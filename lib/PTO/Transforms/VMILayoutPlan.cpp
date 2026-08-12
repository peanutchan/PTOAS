// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutPlan.cpp - First-class VMI layout plan --------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMILayoutPlan.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <functional>

using namespace mlir;
using namespace mlir::pto;

StringRef mlir::pto::layoutPlanAttrName() { return "pto.vmi.layout_plan"; }

namespace {

std::string trim(StringRef s) {
  s = s.trim();
  return s.str();
}

std::string toLower(StringRef s) {
  std::string out = s.str();
  for (char &c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

SmallVector<std::string, 8> splitMorphs(StringRef text) {
  SmallVector<std::string, 8> out;
  SmallVector<StringRef, 8> parts;
  text.split(parts, "->");
  for (StringRef p : parts) {
    std::string t = trim(p);
    if (!t.empty())
      out.push_back(t);
  }
  return out;
}

std::optional<int64_t> parseExpandGroup(StringRef morph) {
  if (!morph.starts_with("Expand("))
    return std::nullopt;
  size_t comma = morph.find(',');
  size_t lpar = morph.find('(');
  if (comma == StringRef::npos || lpar == StringRef::npos)
    return std::nullopt;
  int64_t g = 0;
  if (morph.slice(lpar + 1, comma).trim().getAsInteger(10, g))
    return std::nullopt;
  return g;
}

std::optional<std::string> parseExpandMode(StringRef morph) {
  if (!morph.starts_with("Expand("))
    return std::nullopt;
  size_t comma = morph.find(',');
  size_t rpar = morph.rfind(')');
  if (comma == StringRef::npos || rpar == StringRef::npos || rpar <= comma)
    return std::nullopt;
  return trim(morph.slice(comma + 1, rpar));
}

std::optional<int64_t> parseCompactGroup(StringRef morph) {
  if (!morph.starts_with("Compact("))
    return std::nullopt;
  size_t lpar = morph.find('(');
  size_t rpar = morph.rfind(')');
  if (lpar == StringRef::npos || rpar == StringRef::npos || rpar <= lpar)
    return std::nullopt;
  int64_t g = 0;
  if (morph.slice(lpar + 1, rpar).trim().getAsInteger(10, g))
    return std::nullopt;
  return g;
}

/// Parse MaterializeBoundary[@scope] → scope string (may be empty).
std::optional<std::string> parseMaterializeMorph(StringRef morph) {
  if (!morph.starts_with("MaterializeBoundary"))
    return std::nullopt;
  if (morph == "MaterializeBoundary")
    return std::string();
  if (!morph.starts_with("MaterializeBoundary@"))
    return std::nullopt;
  return trim(morph.drop_front(StringRef("MaterializeBoundary@").size()));
}

/// Parse Split(axis,B)[@scope] or Split(axis,B=N)[@scope].
struct SplitMorph {
  std::string axis;
  int64_t factor = 0;
  std::string scope;
};

std::optional<SplitMorph> parseSplitMorph(StringRef morph) {
  if (!morph.starts_with("Split("))
    return std::nullopt;
  size_t lpar = morph.find('(');
  size_t rpar = morph.find(')');
  if (lpar == StringRef::npos || rpar == StringRef::npos || rpar <= lpar)
    return std::nullopt;
  StringRef args = morph.slice(lpar + 1, rpar).trim();
  auto [axisPart, factorPart] = args.split(',');
  SplitMorph out;
  out.axis = trim(axisPart);
  StringRef f = factorPart.trim();
  if (f.starts_with("B=") || f.starts_with("b="))
    f = f.drop_front(2);
  if (f.getAsInteger(10, out.factor) || out.factor <= 0 || out.axis.empty())
    return std::nullopt;
  if (rpar + 1 < morph.size() && morph[rpar + 1] == '@')
    out.scope = trim(morph.drop_front(rpar + 2));
  return out;
}

bool isValidScope(StringRef scope) {
  return scope.empty() || scope == "vf" || scope == "inner" || scope == "outer" ||
         scope == "stage";
}

struct VMIFacts {
  SmallVector<int64_t, 4> groups;
  bool hasGroupReduce = false;
  bool hasGroupStore = false;
  bool hasGroupVbrc = false;
  bool hasUi8Store = false;
  bool hasMixAccSpill = false; // nested scf.for with load/vmula/store pattern
};

bool isInnerMixAccSpill(scf::ForOp outer, scf::ForOp inner) {
  // Detect: vload(mix) → vmula → vstore(mix) inside inner, offset uses outer IV.
  bool sawLoad = false;
  bool sawVmula = false;
  bool sawStore = false;
  Value mixPtr;
  inner.walk([&](Operation *op) {
    if (auto load = dyn_cast<VMIvLoadOp>(op)) {
      if (!mixPtr)
        mixPtr = load.getSource();
      if (load.getSource() == mixPtr)
        sawLoad = true;
    } else if (isa<VMIVmulaOp>(op)) {
      sawVmula = true;
    } else if (auto store = dyn_cast<VMIvStoreOp>(op)) {
      if (mixPtr && store.getDestination() == mixPtr)
        sawStore = true;
    }
  });
  (void)outer;
  return sawLoad && sawVmula && sawStore;
}

VMIFacts collectFacts(ModuleOp module) {
  VMIFacts facts;
  module.walk([&](Operation *op) {
    if (auto vbrc = dyn_cast<VMIVbrcOp>(op)) {
      if (auto g = vbrc.getGroupAttr()) {
        facts.hasGroupVbrc = true;
        facts.groups.push_back(g.getInt());
      }
    }
    if (auto store = dyn_cast<VMIvStoreOp>(op)) {
      if (auto g = store.getGroupAttr()) {
        facts.hasGroupStore = true;
        facts.groups.push_back(g.getInt());
      }
      if (!store.getValues().empty()) {
        if (auto ty = dyn_cast<VMIVRegType>(store.getValues().front().getType()))
          if (ty.getElementType().isInteger(8) &&
              !ty.getElementType().isSignedInteger(8))
            facts.hasUi8Store = true;
      }
    }
    if (isa<VMIGroupReduceMaxFOp, VMIGroupReduceAddFOp, VMIGroupReduceMinFOp,
            VMIGroupReduceMaxIOp, VMIGroupReduceAddIOp, VMIGroupReduceMinIOp>(
            op)) {
      facts.hasGroupReduce = true;
      if (auto attr = op->getAttrOfType<IntegerAttr>("num_groups"))
        facts.groups.push_back(attr.getInt());
    }
    if (op->getName().getStringRef().contains("vcmax") ||
        op->getName().getStringRef().contains("vcadd") ||
        op->getName().getStringRef().contains("vcmin")) {
      if (auto g = op->getAttrOfType<IntegerAttr>("group")) {
        facts.hasGroupReduce = true;
        facts.groups.push_back(g.getInt());
      }
    }
    if (auto outer = dyn_cast<scf::ForOp>(op)) {
      for (Operation &innerOp : *outer.getBody()) {
        if (auto inner = dyn_cast<scf::ForOp>(&innerOp)) {
          if (isInnerMixAccSpill(outer, inner))
            facts.hasMixAccSpill = true;
        }
      }
    }
  });
  llvm::sort(facts.groups);
  facts.groups.erase(llvm::unique(facts.groups), facts.groups.end());
  return facts;
}

bool morphListContains(ArrayRef<std::string> morphs, StringRef needle) {
  for (const std::string &m : morphs)
    if (StringRef(m).contains_insensitive(needle))
      return true;
  return false;
}

std::string resolveExpand(const VMILayoutPlan &plan) {
  std::string expand = plan.preferredExpand;
  for (const std::string &m : plan.morphs)
    if (auto mode = parseExpandMode(m))
      expand = *mode;
  return expand;
}

int64_t resolveGroup(const VMILayoutPlan &plan) {
  int64_t group = plan.group;
  for (const std::string &m : plan.morphs) {
    if (auto g = parseCompactGroup(m))
      group = *g;
    if (auto g = parseExpandGroup(m))
      group = *g;
  }
  return group;
}

LogicalResult applyMaterializeExpand(ModuleOp module, StringRef expand,
                                     raw_ostream &diag) {
  OpBuilder builder(module.getContext());
  DenseMap<Value, VMIvStoreOp> storesByValue;
  module.walk([&](VMIvStoreOp store) {
    if (!store.getGroupAttr())
      return;
    if (store.getValues().empty())
      return;
    storesByValue[store.getValues().front()] = store;
  });

  SmallVector<VMIVbrcOp> toRewrite;
  module.walk([&](VMIVbrcOp vbrc) {
    if (!vbrc.getGroupAttr())
      return;
    if (!storesByValue.count(vbrc.getValue()))
      return;
    toRewrite.push_back(vbrc);
  });

  int rewriteCount = 0;
  for (VMIVbrcOp vbrc : toRewrite) {
    VMIvStoreOp store = storesByValue.lookup(vbrc.getValue());
    if (!store)
      continue;
    builder.setInsertionPoint(vbrc);
    auto kind = MemBarAttr::get(builder.getContext(), MemBarKind::VST_VLD);
    builder.create<MemBarOp>(vbrc.getLoc(), kind);

    Value stride = store.getStride();
    if (!stride) {
      diag << "layout plan: grouped vstore missing stride for materialize\n";
      return failure();
    }
    auto load = builder.create<VMIvLoadOp>(
        vbrc.getLoc(), TypeRange{vbrc.getResult().getType()},
        store.getDestination(), store.getOffset(), stride,
        /*block_stride=*/Value(), /*repeat_stride=*/Value(),
        builder.getStringAttr("brc"), vbrc.getGroupAttr(),
        /*pmode=*/StringAttr(), builder.getStringAttr(expand));
    vbrc.getResult().replaceAllUsesWith(load.getResult(0));
    vbrc.erase();
    ++rewriteCount;
  }
  if (rewriteCount == 0) {
    diag << "layout plan: no compact-store/group-vbrc pair to materialize\n";
    return failure();
  }
  return success();
}

LogicalResult applyKeepLiveExpandAnnotate(ModuleOp module, StringRef expand,
                                          raw_ostream &diag) {
  int rewriteCount = 0;
  module.walk([&](VMIVbrcOp vbrc) {
    if (!vbrc.getGroupAttr())
      return;
    vbrc->setAttr("preferred_expand",
                  StringAttr::get(vbrc.getContext(), expand));
    ++rewriteCount;
  });
  if (rewriteCount == 0) {
    diag << "layout plan: no grouped vbrc to annotate for KeepLive expand\n";
    return failure();
  }
  return success();
}

/// Find grouped-store → same-SSA grouped-vbrc cut in a flat block.
static bool findStageCutInBlock(Block &block, VMIvStoreOp &cutStore,
                                VMIVbrcOp &cutVbrc) {
  DenseMap<Value, VMIvStoreOp> stores;
  for (Operation &op : block) {
    // Only direct ops; nested vecscopes are handled separately.
    if (isa<VecScopeOp>(&op))
      continue;
    if (auto store = dyn_cast<VMIvStoreOp>(&op)) {
      if (store.getGroupAttr() && !store.getValues().empty())
        stores[store.getValues().front()] = store;
    }
    if (auto vbrc = dyn_cast<VMIVbrcOp>(&op)) {
      if (!vbrc.getGroupAttr())
        continue;
      auto it = stores.find(vbrc.getValue());
      if (it != stores.end() && it->second->isBeforeInBlock(vbrc)) {
        cutStore = it->second;
        cutVbrc = vbrc;
        return true;
      }
    }
  }
  return false;
}

/// Split at compact group-store that feeds a grouped vbrc.
/// Prefer an existing pto.vecscope cut; else invent VF0/membar/VF1 around a
/// flat func-body cut (DSL-lowered VMI often has no vecscope yet).
LogicalResult applyStageSplitVecScope(ModuleOp module, raw_ostream &diag) {
  VecScopeOp targetScope;
  Block *srcBlock = nullptr;
  VMIvStoreOp cutStore;
  VMIVbrcOp cutVbrc;

  module.walk([&](VecScopeOp scope) {
    if (targetScope)
      return;
    if (scope.getBody().empty())
      return;
    VMIvStoreOp store;
    VMIVbrcOp vbrc;
    if (findStageCutInBlock(scope.getBody().front(), store, vbrc)) {
      targetScope = scope;
      srcBlock = &scope.getBody().front();
      cutStore = store;
      cutVbrc = vbrc;
    }
  });

  if (!cutStore) {
    module.walk([&](func::FuncOp func) {
      if (cutStore || func.getBody().empty())
        return;
      VMIvStoreOp store;
      VMIVbrcOp vbrc;
      if (findStageCutInBlock(func.getBody().front(), store, vbrc)) {
        srcBlock = &func.getBody().front();
        cutStore = store;
        cutVbrc = vbrc;
      }
    });
  }

  if (!cutStore || !srcBlock) {
    diag << "layout plan: StageSplit/Materialize@stage found no "
            "grouped-store→grouped-vbrc cut (inside pto.vecscope or flat "
            "func body)\n";
    return failure();
  }

  Value stride = cutStore.getStride();
  if (!stride) {
    diag << "layout plan: StageSplit compact store missing stride\n";
    return failure();
  }

  OpBuilder builder(module.getContext());
  Location loc = cutStore.getLoc();
  if (targetScope)
    builder.setInsertionPoint(targetScope);
  else
    builder.setInsertionPointToStart(srcBlock);

  auto ensureBody = [](VecScopeOp scope) -> Block & {
    if (scope.getBody().empty())
      scope.getBody().push_back(new Block());
    return scope.getBody().front();
  };

  auto vf0 = builder.create<VecScopeOp>(loc);
  auto kind = MemBarAttr::get(builder.getContext(), MemBarKind::VST_VLD);
  auto memBar = builder.create<MemBarOp>(loc, kind);
  auto vf1 = builder.create<VecScopeOp>(loc);

  Block &b0 = ensureBody(vf0);
  Block &b1 = ensureBody(vf1);
  Block &src = *srcBlock;

  // Snapshot ops to clone before we mutate the block.
  SmallVector<Operation *> srcOps;
  if (targetScope) {
    for (Operation &op : targetScope.getBody().front())
      srcOps.push_back(&op);
  } else {
    for (Operation &op : src) {
      if (&op == vf0.getOperation() || &op == vf1.getOperation() ||
          &op == memBar.getOperation())
        continue;
      // Keep func terminator outside the invented VF scopes.
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      srcOps.push_back(&op);
    }
  }

  IRMapping map0;
  OpBuilder b0b(&b0, b0.end());
  bool pastCut = false;
  for (Operation *opPtr : srcOps) {
    Operation &op = *opPtr;
    if (!pastCut) {
      b0b.clone(op, map0);
      if (&op == cutStore.getOperation())
        pastCut = true;
    }
  }

  IRMapping map1;
  OpBuilder b1b(&b1, b1.end());
  bool reachedCut = false;
  for (Operation *opPtr : srcOps) {
    Operation &op = *opPtr;
    if (&op == cutStore.getOperation()) {
      reachedCut = true;
      continue;
    }
    if (&op == cutVbrc.getOperation()) {
      Value dst = map1.lookupOrDefault(cutStore.getDestination());
      Value off = map1.lookupOrDefault(cutStore.getOffset());
      Value st = map1.lookupOrDefault(stride);
      auto load = b1b.create<VMIvLoadOp>(
          cutVbrc.getLoc(), TypeRange{cutVbrc.getResult().getType()}, dst, off,
          st, Value(), Value(), b1b.getStringAttr("brc"),
          cutVbrc.getGroupAttr(), StringAttr(), b1b.getStringAttr("e2b"));
      map1.map(cutVbrc.getResult(), load.getResult(0));
      reachedCut = true;
      continue;
    }
    b1b.clone(op, map1);
    (void)reachedCut;
  }

  if (targetScope) {
    targetScope.erase();
  } else {
    // Remove original flat body ops now covered by VF0/VF1.
    for (Operation *op : llvm::reverse(srcOps))
      op->erase();
  }
  return success();
}

/// Hoist mix-acc spill out of inner chunk loop (KeepLive / ASC-like).
LogicalResult applyMhcKeepLive(ModuleOp module, raw_ostream &diag) {
  SmallVector<scf::ForOp> outers;
  module.walk([&](scf::ForOp outer) {
    for (Operation &op : *outer.getBody()) {
      if (auto inner = dyn_cast<scf::ForOp>(&op)) {
        if (isInnerMixAccSpill(outer, inner))
          outers.push_back(outer);
      }
    }
  });
  if (outers.empty()) {
    diag << "layout plan: KeepLive residency found no MHC mix-acc spill nest "
            "(scf.for i_mhc × scf.for chunk with vload/vmula/vstore)\n";
    return failure();
  }

  auto dependsOnlyOnOuter = [&](Value v, scf::ForOp outer, scf::ForOp inner) {
    SmallVector<Value> work = {v};
    DenseSet<Value> seen;
    while (!work.empty()) {
      Value cur = work.pop_back_val();
      if (!seen.insert(cur).second)
        continue;
      if (cur == inner.getInductionVar())
        return false;
      for (Value arg : inner.getRegionIterArgs())
        if (cur == arg)
          return false;
      Operation *def = cur.getDefiningOp();
      if (!def)
        continue; // block arg from outer/func — ok
      if (def->getBlock() == outer.getBody() ||
          def->getParentOp() == outer->getParentOp() ||
          def->getBlock() != inner.getBody()) {
        // Defined outside inner — ok if its operands also ok.
        continue;
      }
      for (Value operand : def->getOperands())
        work.push_back(operand);
    }
    return true;
  };

  for (scf::ForOp outer : outers) {
    scf::ForOp inner;
    for (Operation &op : *outer.getBody()) {
      if (auto cand = dyn_cast<scf::ForOp>(&op)) {
        if (isInnerMixAccSpill(outer, cand)) {
          inner = cand;
          break;
        }
      }
    }
    if (!inner)
      continue;

    VMIvLoadOp spillLoad;
    VMIVmulaOp spillVmula;
    VMIvStoreOp spillStore;
    inner.walk([&](Operation *op) {
      if (auto load = dyn_cast<VMIvLoadOp>(op)) {
        if (!spillLoad)
          spillLoad = load;
      } else if (auto vmula = dyn_cast<VMIVmulaOp>(op)) {
        spillVmula = vmula;
      } else if (auto store = dyn_cast<VMIvStoreOp>(op)) {
        if (spillLoad && store.getDestination() == spillLoad.getSource())
          spillStore = store;
      }
    });
    if (!spillLoad || !spillVmula || !spillStore) {
      diag << "layout plan: MHC KeepLive could not locate spill trio\n";
      return failure();
    }

    // Offset must not depend on the chunk IV.
    if (!dependsOnlyOnOuter(spillLoad.getOffset(), outer, inner) ||
        !dependsOnlyOnOuter(spillStore.getOffset(), outer, inner)) {
      diag << "layout plan: MHC KeepLive mix offset depends on inner chunk IV\n";
      return failure();
    }

    OpBuilder builder(outer.getContext());
    builder.setInsertionPoint(inner);
    IRMapping hoistMap;
    // Clone ops inside inner that compute the mix offset (depend only on outer).
    for (Operation &op : *inner.getBody()) {
      if (isa<scf::YieldOp>(op))
        continue;
      bool usedBySpill = false;
      for (OpOperand &use : op.getUses()) {
        Operation *user = use.getOwner();
        if (user == spillLoad.getOperation() || user == spillStore.getOperation())
          usedBySpill = true;
      }
      // Also hoist if all results are used only by offset chain toward spill.
      if (!usedBySpill) {
        for (Value res : op.getResults()) {
          if (res == spillLoad.getOffset() || res == spillStore.getOffset())
            usedBySpill = true;
        }
      }
      if (!usedBySpill && &op != spillLoad.getOperation() &&
          &op != spillStore.getOperation()) {
        // Hoist defining chain of offset.
        bool definesOffset = false;
        SmallVector<Operation *> stack = {spillLoad.getOffset().getDefiningOp(),
                                          spillStore.getOffset().getDefiningOp()};
        while (!stack.empty()) {
          Operation *cur = stack.pop_back_val();
          if (!cur)
            continue;
          if (cur == &op) {
            definesOffset = true;
            break;
          }
          if (cur->getBlock() != inner.getBody())
            continue;
          for (Value operand : cur->getOperands())
            if (Operation *odef = operand.getDefiningOp())
              stack.push_back(odef);
        }
        if (!definesOffset)
          continue;
      }
      if (&op == spillLoad.getOperation() || &op == spillStore.getOperation() ||
          &op == spillVmula.getOperation() || isa<MemBarOp>(op))
        continue;
      if (hoistMap.contains(op.getResult(0)))
        continue;
      // Clone in dominance order by walking body linearly.
    }
    // Linear hoist of offset producers.
    for (Operation &op : *inner.getBody()) {
      if (isa<scf::YieldOp, MemBarOp>(op) || &op == spillLoad.getOperation() ||
          &op == spillStore.getOperation() || &op == spillVmula.getOperation())
        continue;
      // Skip ops that use inner IV or loads of x/og (chunk-varying).
      bool chunkVarying = false;
      for (Value operand : op.getOperands()) {
        if (operand == inner.getInductionVar())
          chunkVarying = true;
        if (auto load = operand.getDefiningOp<VMIvLoadOp>())
          if (load != spillLoad)
            chunkVarying = true;
      }
      if (isa<VMIvLoadOp>(op) && &op != spillLoad.getOperation())
        chunkVarying = true;
      if (isa<VMIVmulaOp>(op))
        chunkVarying = true;
      if (chunkVarying)
        continue;
      builder.clone(op, hoistMap);
    }

    Operation *newLoad = builder.clone(*spillLoad.getOperation(), hoistMap);
    Value initAcc = newLoad->getResult(0);

    SmallVector<Value> newInitArgs = llvm::to_vector(inner.getInitArgs());
    newInitArgs.push_back(initAcc);

    auto newInner = builder.create<scf::ForOp>(
        inner.getLoc(), inner.getLowerBound(), inner.getUpperBound(),
        inner.getStep(), newInitArgs);

    IRMapping bodyMap;
    bodyMap.map(inner.getInductionVar(), newInner.getInductionVar());
    for (auto [oldA, newA] :
         llvm::zip(inner.getRegionIterArgs(),
                   llvm::drop_end(newInner.getRegionIterArgs())))
      bodyMap.map(oldA, newA);
    // Offset producers already in hoistMap / outer.
    for (auto &entry : hoistMap.getValueMap())
      bodyMap.map(entry.first, entry.second);
    Value newAccArg = newInner.getRegionIterArgs().back();

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(newInner.getBody());
    Value yieldedAcc = newAccArg;
    for (Operation &op : *inner.getBody()) {
      if (isa<scf::YieldOp>(op))
        continue;
      if (&op == spillLoad.getOperation()) {
        bodyMap.map(spillLoad.getResult(0), newAccArg);
        continue;
      }
      if (&op == spillStore.getOperation()) {
        if (!spillStore.getValues().empty()) {
          Value stored = spillStore.getValues().front();
          if (auto mapped = bodyMap.lookupOrNull(stored))
            yieldedAcc = mapped;
          else
            yieldedAcc = stored;
        }
        continue;
      }
      if (isa<MemBarOp>(op))
        continue;
      // Skip already-hoisted offset ops.
      if (!op.getResults().empty() && hoistMap.lookupOrNull(op.getResult(0)))
        continue;
      Operation *cloned = bodyBuilder.clone(op, bodyMap);
      if (&op == spillVmula.getOperation() && !cloned->getResults().empty()) {
        bodyMap.map(spillVmula.getResult(), cloned->getResult(0));
        yieldedAcc = cloned->getResult(0);
      }
    }
    SmallVector<Value> yieldVals;
    if (auto yield = dyn_cast<scf::YieldOp>(inner.getBody()->getTerminator())) {
      for (Value v : yield.getOperands()) {
        if (auto m = bodyMap.lookupOrNull(v))
          yieldVals.push_back(m);
        else
          yieldVals.push_back(v);
      }
    }
    yieldVals.push_back(yieldedAcc);
    bodyBuilder.create<scf::YieldOp>(inner.getLoc(), yieldVals);

    builder.setInsertionPointAfter(newInner);
    Value finalAcc = newInner.getResults().back();
    IRMapping storeMap = hoistMap;
    storeMap.map(spillStore.getValues().front(), finalAcc);
    builder.clone(*spillStore.getOperation(), storeMap);

    if (!inner.getResults().empty()) {
      for (auto [oldR, newR] : llvm::zip(inner.getResults(),
                                         llvm::drop_end(newInner.getResults())))
        oldR.replaceAllUsesWith(newR);
    }
    inner.erase();
  }
  return success();
}

/// Validate Materialize@inner: ensure spill nest exists; insert mem_bar if missing.
LogicalResult applyMhcMaterializeInner(ModuleOp module, raw_ostream &diag) {
  bool found = false;
  module.walk([&](scf::ForOp outer) {
    for (Operation &op : *outer.getBody()) {
      if (auto inner = dyn_cast<scf::ForOp>(&op)) {
        if (!isInnerMixAccSpill(outer, inner))
          continue;
        found = true;
        bool hasBar = false;
        inner.walk([&](MemBarOp) { hasBar = true; });
        if (!hasBar) {
          // Insert mem_bar after each mix store.
          OpBuilder builder(inner.getContext());
          inner.walk([&](VMIvStoreOp store) {
            builder.setInsertionPointAfter(store);
            auto kind =
                MemBarAttr::get(builder.getContext(), MemBarKind::VST_VLD);
            builder.create<MemBarOp>(store.getLoc(), kind);
          });
        }
      }
    }
  });
  if (!found) {
    diag << "layout plan: MaterializeBoundary@inner found no MHC mix-acc "
            "spill nest\n";
    return failure();
  }
  return success();
}

LogicalResult applyMhcSplitOuter(ModuleOp module, const VMILayoutPlan &plan,
                                 raw_ostream &diag) {
  if (plan.splitAxis != "i_mhc") {
    diag << "layout plan: Split currently supports axis 'i_mhc' only (got '"
         << plan.splitAxis << "')\n";
    return failure();
  }
  int64_t B = plan.splitFactor;
  if (B <= 0) {
    diag << "layout plan: Split requires positive factor B\n";
    return failure();
  }

  // First hoist inner spills to KeepLive-at-i_mhc (store once per i_mhc).
  if (failed(applyMhcKeepLive(module, diag)))
    return failure();

  // Then factor the i_mhc loop into outer × inner(B) with one mem_bar per outer.
  scf::ForOp mhcLoop;
  module.walk([&](scf::ForOp outer) {
    if (mhcLoop)
      return;
    // After KeepLive, outer body should have: load, inner chunk for, store.
    // Identify by: contains nested scf.for AND a vstore of mix after it.
    bool hasInner = false;
    bool hasStore = false;
    for (Operation &op : *outer.getBody()) {
      if (isa<scf::ForOp>(op))
        hasInner = true;
      if (isa<VMIvStoreOp>(op))
        hasStore = true;
    }
    if (hasInner && hasStore)
      mhcLoop = outer;
  });
  if (!mhcLoop) {
    diag << "layout plan: Split(i_mhc) could not find MHC loop after KeepLive "
            "hoist\n";
    return failure();
  }

  // Require constant trip count divisible by B.
  auto ub = mhcLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto lb = mhcLoop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = mhcLoop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!ub || !lb || !step || step.value() != 1 || lb.value() != 0) {
    diag << "layout plan: Split(i_mhc) requires scf.for 0..N step 1 with "
            "constant bounds\n";
    return failure();
  }
  int64_t N = ub.value();
  if (N % B != 0) {
    diag << "layout plan: Split(i_mhc," << B << ") requires MHC=" << N
         << " divisible by B\n";
    return failure();
  }
  int64_t nOuter = N / B;

  OpBuilder builder(mhcLoop.getContext());
  Location loc = mhcLoop.getLoc();
  builder.setInsertionPoint(mhcLoop);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value cB = builder.create<arith::ConstantIndexOp>(loc, B);
  Value cOuter = builder.create<arith::ConstantIndexOp>(loc, nOuter);

  auto newOuter = builder.create<scf::ForOp>(loc, c0, cOuter, c1);
  OpBuilder ob = OpBuilder::atBlockBegin(newOuter.getBody());
  Value outerIv = newOuter.getInductionVar();

  // Inner ii loop 0..B; i_mhc = outer*B + ii.
  auto newInnerMhc = ob.create<scf::ForOp>(loc, c0, cB, c1);
  OpBuilder ib = OpBuilder::atBlockBegin(newInnerMhc.getBody());
  Value ii = newInnerMhc.getInductionVar();
  Value outerScaled = ib.create<arith::MulIOp>(loc, outerIv, cB);
  Value iMhc = ib.create<arith::AddIOp>(loc, outerScaled, ii);

  // Clone original MHC body with IV remapped to iMhc.
  IRMapping map;
  map.map(mhcLoop.getInductionVar(), iMhc);
  Operation *clonedStore = nullptr;
  for (Operation &op : *mhcLoop.getBody()) {
    if (isa<scf::YieldOp>(op))
      continue;
    Operation *cloned = ib.clone(op, map);
    if (isa<VMIvStoreOp>(cloned))
      clonedStore = cloned;
  }

  // Move mem_bar to after the ii loop (once per outer tile), not per ii.
  // Remove bars cloned inside and insert one after newInnerMhc.
  newInnerMhc.walk([&](MemBarOp bar) {
    if (bar->getParentOp() == newInnerMhc.getOperation() ||
        bar->getParentOfType<scf::ForOp>() == newInnerMhc)
      return;
  });
  // Erase mem_bars inside the ii body (including nested chunk loops).
  SmallVector<MemBarOp> bars;
  newInnerMhc.walk([&](MemBarOp bar) { bars.push_back(bar); });
  for (MemBarOp bar : bars)
    bar.erase();

  ob.setInsertionPointAfter(newInnerMhc);
  auto kind = MemBarAttr::get(ob.getContext(), MemBarKind::VST_VLD);
  ob.create<MemBarOp>(loc, kind);
  (void)clonedStore;

  mhcLoop.erase();
  return success();
}

} // namespace

VMIResidencyKind mlir::pto::VMILayoutPlan::residencyKind() const {
  if (residency == "keeplive")
    return VMIResidencyKind::KeepLive;
  if (residency == "materialize")
    return VMIResidencyKind::Materialize;
  if (residency == "split")
    return VMIResidencyKind::Split;
  return VMIResidencyKind::Unspecified;
}

void mlir::pto::normalizeLayoutPlan(VMILayoutPlan &plan) {
  bool hasKeep = false;
  bool hasMat = false;
  bool hasSplit = false;
  bool hasStage = plan.stageSplit;

  for (const std::string &m : plan.morphs) {
    if (StringRef(m).equals_insensitive("KeepLive") ||
        StringRef(m).starts_with_insensitive("KeepLive@"))
      hasKeep = true;
    if (auto scope = parseMaterializeMorph(m)) {
      hasMat = true;
      if (!scope->empty() && plan.materializeScope.empty())
        plan.materializeScope = *scope;
    }
    if (auto split = parseSplitMorph(m)) {
      hasSplit = true;
      if (plan.splitAxis.empty())
        plan.splitAxis = split->axis;
      if (!plan.splitFactor)
        plan.splitFactor = split->factor;
      if (!split->scope.empty() && plan.materializeScope.empty())
        plan.materializeScope = split->scope;
    }
    if (StringRef(m).starts_with("StageSplit")) {
      hasStage = true;
      plan.stageSplit = true;
      if (plan.materializeScope.empty())
        plan.materializeScope = "stage";
    }
  }

  if (plan.materializeBoundary)
    hasMat = true;

  // Explicit residency key wins; else derive.
  if (plan.residency.empty()) {
    if (hasSplit)
      plan.residency = "split";
    else if (hasStage || hasMat)
      plan.residency = "materialize";
    else if (hasKeep)
      plan.residency = "keeplive";
  }

  if (plan.residency == "materialize") {
    plan.materializeBoundary = true;
    if (plan.materializeScope.empty())
      plan.materializeScope = plan.stageSplit ? "stage" : "vf";
    if (plan.materializeScope == "stage")
      plan.stageSplit = true;
  }
  if (plan.residency == "split") {
    plan.materializeBoundary = true;
    if (plan.materializeScope.empty())
      plan.materializeScope = "outer";
  }
  if (plan.residency == "keeplive") {
    plan.materializeBoundary = false;
    plan.stageSplit = false;
  }

  // Alias: @inner on expand-only kernels ≡ @vf.
  if (plan.materializeScope == "inner" && plan.residency == "materialize" &&
      plan.splitAxis.empty()) {
    // Keep "inner" for MHC; Q-VMI path treats inner as vf when no mix spill.
  }
}

FailureOr<VMILayoutPlan>
mlir::pto::parseLayoutPlanText(StringRef text, raw_ostream &diag) {
  VMILayoutPlan plan;
  plan.source = "override";
  for (StringRef line : llvm::split(text, '\n')) {
    line = line.trim();
    if (line.empty() || line.starts_with("#") || line.starts_with("//"))
      continue;
    if (line.starts_with("layout.algebra") || line == "{" || line == "}" ||
        line.starts_with("input "))
      continue;
    auto [key, value] = line.split(':');
    if (value.empty()) {
      plan.morphs.push_back(trim(line));
      continue;
    }
    StringRef k = key.trim();
    StringRef v = value.trim();
    if (k.equals_insensitive("version")) {
      if (v.getAsInteger(10, plan.version)) {
        diag << "layout plan: invalid version\n";
        return failure();
      }
    } else if (k.equals_insensitive("source")) {
      plan.source = trim(v);
    } else if (k.equals_insensitive("preferred_expand")) {
      plan.preferredExpand = trim(v);
    } else if (k.equals_insensitive("group")) {
      if (v.getAsInteger(10, plan.group)) {
        diag << "layout plan: invalid group\n";
        return failure();
      }
    } else if (k.equals_insensitive("materialize_boundary")) {
      plan.materializeBoundary = v.equals_insensitive("true") || v == "1";
    } else if (k.equals_insensitive("stage_split")) {
      plan.stageSplit = v.equals_insensitive("true") || v == "1";
    } else if (k.equals_insensitive("residency")) {
      plan.residency = toLower(trim(v));
    } else if (k.equals_insensitive("materialize_scope")) {
      plan.materializeScope = toLower(trim(v));
    } else if (k.equals_insensitive("split_axis")) {
      plan.splitAxis = trim(v);
    } else if (k.equals_insensitive("split_factor")) {
      if (v.getAsInteger(10, plan.splitFactor)) {
        diag << "layout plan: invalid split_factor\n";
        return failure();
      }
    } else if (k.equals_insensitive("morphs")) {
      plan.morphs = splitMorphs(v);
    } else {
      diag << "layout plan: unknown key '" << k << "'\n";
      return failure();
    }
  }

  for (const std::string &m : plan.morphs) {
    if (auto g = parseCompactGroup(m))
      if (!plan.group)
        plan.group = *g;
    if (auto g = parseExpandGroup(m))
      if (!plan.group)
        plan.group = *g;
    if (auto mode = parseExpandMode(m))
      if (plan.preferredExpand.empty())
        plan.preferredExpand = *mode;
  }
  normalizeLayoutPlan(plan);
  if (plan.source.empty())
    plan.source = "override";
  return plan;
}

FailureOr<VMILayoutPlan>
mlir::pto::parseLayoutPlanAttr(DictionaryAttr attr, raw_ostream &diag) {
  if (!attr) {
    diag << "layout plan attribute missing\n";
    return failure();
  }
  VMILayoutPlan plan;
  if (auto v = attr.getAs<IntegerAttr>("version"))
    plan.version = v.getInt();
  if (auto s = attr.getAs<StringAttr>("source"))
    plan.source = s.getValue().str();
  if (auto s = attr.getAs<StringAttr>("preferred_expand"))
    plan.preferredExpand = s.getValue().str();
  if (auto g = attr.getAs<IntegerAttr>("group"))
    plan.group = g.getInt();
  if (auto b = attr.getAs<BoolAttr>("materialize_boundary"))
    plan.materializeBoundary = b.getValue();
  if (auto b = attr.getAs<BoolAttr>("stage_split"))
    plan.stageSplit = b.getValue();
  if (auto s = attr.getAs<StringAttr>("residency"))
    plan.residency = s.getValue().str();
  if (auto s = attr.getAs<StringAttr>("materialize_scope"))
    plan.materializeScope = s.getValue().str();
  if (auto s = attr.getAs<StringAttr>("split_axis"))
    plan.splitAxis = s.getValue().str();
  if (auto f = attr.getAs<IntegerAttr>("split_factor"))
    plan.splitFactor = f.getInt();
  if (auto arr = attr.getAs<ArrayAttr>("morphs")) {
    for (Attribute a : arr)
      if (auto s = dyn_cast<StringAttr>(a))
        plan.morphs.push_back(s.getValue().str());
  }
  normalizeLayoutPlan(plan);
  return plan;
}

DictionaryAttr mlir::pto::encodeLayoutPlanAttr(MLIRContext *ctx,
                                               const VMILayoutPlan &plan) {
  SmallVector<NamedAttribute, 12> fields;
  Builder b(ctx);
  fields.emplace_back(b.getStringAttr("version"),
                      b.getI64IntegerAttr(plan.version));
  fields.emplace_back(b.getStringAttr("source"), b.getStringAttr(plan.source));
  if (!plan.preferredExpand.empty())
    fields.emplace_back(b.getStringAttr("preferred_expand"),
                        b.getStringAttr(plan.preferredExpand));
  if (plan.group)
    fields.emplace_back(b.getStringAttr("group"),
                        b.getI64IntegerAttr(plan.group));
  fields.emplace_back(b.getStringAttr("materialize_boundary"),
                      b.getBoolAttr(plan.materializeBoundary));
  fields.emplace_back(b.getStringAttr("stage_split"),
                      b.getBoolAttr(plan.stageSplit));
  if (!plan.residency.empty())
    fields.emplace_back(b.getStringAttr("residency"),
                        b.getStringAttr(plan.residency));
  if (!plan.materializeScope.empty())
    fields.emplace_back(b.getStringAttr("materialize_scope"),
                        b.getStringAttr(plan.materializeScope));
  if (!plan.splitAxis.empty())
    fields.emplace_back(b.getStringAttr("split_axis"),
                        b.getStringAttr(plan.splitAxis));
  if (plan.splitFactor)
    fields.emplace_back(b.getStringAttr("split_factor"),
                        b.getI64IntegerAttr(plan.splitFactor));
  SmallVector<Attribute, 8> morphAttrs;
  for (const std::string &m : plan.morphs)
    morphAttrs.push_back(b.getStringAttr(m));
  fields.emplace_back(b.getStringAttr("morphs"), b.getArrayAttr(morphAttrs));
  return DictionaryAttr::get(ctx, fields);
}

std::string mlir::pto::formatLayoutPlanText(const VMILayoutPlan &plan) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "version: " << plan.version << "\n";
  os << "source: " << plan.source << "\n";
  if (!plan.preferredExpand.empty())
    os << "preferred_expand: " << plan.preferredExpand << "\n";
  if (plan.group)
    os << "group: " << plan.group << "\n";
  os << "materialize_boundary: "
     << (plan.materializeBoundary ? "true" : "false") << "\n";
  os << "stage_split: " << (plan.stageSplit ? "true" : "false") << "\n";
  if (!plan.residency.empty())
    os << "residency: " << plan.residency << "\n";
  if (!plan.materializeScope.empty())
    os << "materialize_scope: " << plan.materializeScope << "\n";
  if (!plan.splitAxis.empty())
    os << "split_axis: " << plan.splitAxis << "\n";
  if (plan.splitFactor)
    os << "split_factor: " << plan.splitFactor << "\n";
  os << "morphs:";
  for (size_t i = 0; i < plan.morphs.size(); ++i) {
    os << (i == 0 ? " " : " -> ");
    os << plan.morphs[i];
  }
  os << "\n";
  return out;
}

FailureOr<VMILayoutPlan> mlir::pto::inferLayoutPlan(ModuleOp module,
                                                    raw_ostream &diag) {
  VMIFacts facts = collectFacts(module);
  VMILayoutPlan plan;
  plan.source = "auto";
  plan.version = 1;
  if (!facts.groups.empty())
    plan.group = facts.groups.back();

  // Soft-cost auto selection (must agree with explorer auto_infer.py):
  //   KeepLive+vbrc ≪ Materialize@vf+e2b ≪ StageSplit+e2b
  //   MHC: KeepLive ≪ Split@outer ≪ Materialize@inner (membar×trips)

  if (facts.hasMixAccSpill && !facts.hasGroupVbrc) {
    // Prefer KeepLive when VRF can hold one mix-acc tile (v1 default).
    plan.residency = "keeplive";
    plan.materializeBoundary = false;
    plan.morphs = {"KeepLive"};
    normalizeLayoutPlan(plan);
    (void)diag;
    return plan;
  }

  if (!facts.hasGroupReduce && !facts.hasGroupVbrc && !facts.hasGroupStore) {
    plan.residency = "keeplive";
    plan.morphs = {"KeepLive"};
    normalizeLayoutPlan(plan);
    return plan;
  }

  if (plan.group)
    plan.morphs.push_back(("Compact(" + std::to_string(plan.group) + ")"));

  if (facts.hasGroupVbrc) {
    // Soft-cost champion: KeepLive + vbrc (no UB reload / no VF split).
    plan.residency = "keeplive";
    plan.preferredExpand = "vbrc";
    plan.morphs.push_back("Expand(" + std::to_string(plan.group) + ",vbrc)");
    plan.morphs.push_back("KeepLive");
  } else if (facts.hasGroupStore && facts.hasGroupReduce) {
    // Compact store without vbrc yet — Materialize@vf + e2b is the Legal path
    // once a broadcast consumer is introduced; stamp intent for assignment.
    plan.residency = "materialize";
    plan.materializeScope = "vf";
    plan.materializeBoundary = true;
    plan.preferredExpand = "e2b";
    plan.morphs.push_back("MaterializeBoundary@vf");
    plan.morphs.push_back("Expand(" + std::to_string(plan.group) + ",e2b)");
  }
  if (facts.hasUi8Store)
    plan.morphs.push_back("Pack(PK4)");
  normalizeLayoutPlan(plan);
  (void)diag;
  return plan;
}

LogicalResult mlir::pto::validateLayoutPlan(ModuleOp module,
                                            const VMILayoutPlan &planIn,
                                            raw_ostream &diag) {
  VMILayoutPlan plan = planIn;
  normalizeLayoutPlan(plan);

  if (morphListContains(plan.morphs, "masked_recompute")) {
    diag << "layout plan: Expand(masked_recompute) has no lowering\n";
    return failure();
  }
  if (morphListContains(plan.morphs, "proposed")) {
    diag << "layout plan: proposed HW morphs are not current-ISA\n";
    return failure();
  }

  // Count conflicting residency morphs.
  int residencyMorphs = 0;
  for (const std::string &m : plan.morphs) {
    if (parseSplitMorph(m))
      ++residencyMorphs;
    else if (parseMaterializeMorph(m))
      ++residencyMorphs;
    else if (StringRef(m).starts_with("StageSplit"))
      ++residencyMorphs;
  }
  // KeepLive may coexist as annotation with expand; Split/Materialize/Stage are exclusive.
  if (plan.residency == "split" && plan.residencyKind() == VMIResidencyKind::Split) {
    if (plan.splitAxis.empty() || plan.splitFactor <= 0) {
      diag << "layout plan: residency=split requires split_axis and "
              "split_factor (or Split(axis,B) morph)\n";
      return failure();
    }
  }
  if (!isValidScope(plan.materializeScope)) {
    diag << "layout plan: invalid materialize_scope '" << plan.materializeScope
         << "' (expected vf|inner|outer|stage)\n";
    return failure();
  }

  std::string expand = resolveExpand(plan);
  int64_t group = resolveGroup(plan);

  if (expand == "brc" && group != 1) {
    diag << "layout plan: Expand(" << group
         << ",brc)/BRC_BLK is not lowered in current PTOAS\n";
    return failure();
  }

  VMIFacts facts = collectFacts(module);
  VMIResidencyKind kind = plan.residencyKind();

  // MHC residency paths.
  if (kind == VMIResidencyKind::KeepLive && facts.hasMixAccSpill &&
      !facts.hasGroupVbrc) {
    return success();
  }
  if (kind == VMIResidencyKind::Materialize &&
      (plan.materializeScope == "inner" || plan.materializeScope == "outer") &&
      facts.hasMixAccSpill) {
    return success();
  }
  if (kind == VMIResidencyKind::Split) {
    if (!facts.hasMixAccSpill) {
      diag << "layout plan: Split requires an MHC mix-acc spill nest in VMI\n";
      return failure();
    }
    return success();
  }

  // StageSplit / @stage requires a cut point (vecscope or flat func body).
  if (plan.stageSplit || plan.materializeScope == "stage") {
    bool hasCut = false;
    module.walk([&](VecScopeOp scope) {
      if (scope.getBody().empty())
        return;
      VMIvStoreOp store;
      VMIVbrcOp vbrc;
      if (findStageCutInBlock(scope.getBody().front(), store, vbrc))
        hasCut = true;
    });
    if (!hasCut) {
      module.walk([&](func::FuncOp func) {
        if (func.getBody().empty())
          return;
        VMIvStoreOp store;
        VMIVbrcOp vbrc;
        if (findStageCutInBlock(func.getBody().front(), store, vbrc))
          hasCut = true;
      });
    }
    if (!hasCut) {
      diag << "layout plan: StageSplit/Materialize@stage requires "
              "grouped-store→grouped-vbrc inside pto.vecscope or flat "
              "func body\n";
      return failure();
    }
  }

  bool needMaterialize =
      kind == VMIResidencyKind::Materialize || plan.materializeBoundary ||
      expand == "e2b" || expand == "brc";
  // KeepLive + register expand must not require materialize.
  if (kind == VMIResidencyKind::KeepLive &&
      (expand == "vbrc" || expand == "vselr"))
    needMaterialize = false;

  if ((expand == "e2b" || expand == "brc") && kind == VMIResidencyKind::KeepLive) {
    diag << "layout plan: Expand(" << expand
         << ") is incompatible with residency=keeplive (needs materialize)\n";
    return failure();
  }

  if (group && !facts.groups.empty()) {
    bool match = llvm::is_contained(facts.groups, group);
    if (!match) {
      diag << "layout plan: Compact/Expand(G=" << group
           << ") does not match VMI groups {";
      llvm::interleaveComma(facts.groups, diag);
      diag << "}\n";
      return failure();
    }
  }

  if (needMaterialize && expand == "e2b") {
    if (!facts.hasGroupStore || !facts.hasGroupVbrc) {
      diag << "layout plan: MaterializeBoundary+Expand(e2b) requires a grouped "
              "vstore and grouped vbrc in the VMI\n";
      return failure();
    }
  }
  if ((expand == "vbrc" || expand == "vselr") && !facts.hasGroupVbrc &&
      kind != VMIResidencyKind::KeepLive) {
    // KeepLive without vbrc may be MHC path (already handled).
    if (!facts.hasMixAccSpill) {
      diag << "layout plan: Expand(" << expand
           << ") requires a grouped pto.vmi.vbrc\n";
      return failure();
    }
  }
  if ((expand == "vbrc" || expand == "vselr") && facts.hasGroupVbrc) {
    // ok
  }
  (void)residencyMorphs;
  return success();
}

LogicalResult mlir::pto::applyLayoutPlan(ModuleOp module,
                                         const VMILayoutPlan &planIn,
                                         raw_ostream &diag) {
  VMILayoutPlan plan = planIn;
  normalizeLayoutPlan(plan);
  if (failed(validateLayoutPlan(module, plan, diag)))
    return failure();

  std::string expand = resolveExpand(plan);
  VMIResidencyKind kind = plan.residencyKind();
  VMIFacts facts = collectFacts(module);

  // --- MHC reducer residency ---
  if (facts.hasMixAccSpill &&
      (kind == VMIResidencyKind::KeepLive || kind == VMIResidencyKind::Split ||
       (kind == VMIResidencyKind::Materialize &&
        (plan.materializeScope == "inner" || plan.materializeScope == "outer")))) {
    if (kind == VMIResidencyKind::KeepLive) {
      if (failed(applyMhcKeepLive(module, diag)))
        return failure();
    } else if (kind == VMIResidencyKind::Split) {
      if (failed(applyMhcSplitOuter(module, plan, diag)))
        return failure();
    } else if (plan.materializeScope == "inner") {
      if (failed(applyMhcMaterializeInner(module, diag)))
        return failure();
    } else if (plan.materializeScope == "outer") {
      // outer without Split: hoist to KeepLive-at-i_mhc then bar once per i_mhc.
      if (failed(applyMhcKeepLive(module, diag)))
        return failure();
      // Ensure mem_bar after each i_mhc store.
      module.walk([&](scf::ForOp outer) {
        OpBuilder builder(outer.getContext());
        for (Operation &op : llvm::make_early_inc_range(*outer.getBody())) {
          if (auto store = dyn_cast<VMIvStoreOp>(&op)) {
            bool hasBarAfter = isa_and_nonnull<MemBarOp>(store->getNextNode());
            if (!hasBarAfter) {
              builder.setInsertionPointAfter(store);
              auto kindAttr =
                  MemBarAttr::get(builder.getContext(), MemBarKind::VST_VLD);
              builder.create<MemBarOp>(store.getLoc(), kindAttr);
            }
          }
        }
      });
    }
  } else {
    // --- Q-VMI expand / stage paths ---
    if (plan.stageSplit || plan.materializeScope == "stage") {
      if (failed(applyStageSplitVecScope(module, diag)))
        return failure();
      // After split, materialize expand in VF1 if requested.
      if (expand == "e2b" || expand == "brc") {
        // StageSplit remat already inserted expand load; optional second pass
        // for any remaining store+vbrc pairs.
        // Ignore failure if already rematerialized.
        std::string unused;
        llvm::raw_string_ostream sink(unused);
        (void)applyMaterializeExpand(module, expand, sink);
      }
    } else if (kind == VMIResidencyKind::KeepLive ||
               (!plan.materializeBoundary &&
                (expand == "vbrc" || expand == "vselr"))) {
      if (!expand.empty() && (expand == "vbrc" || expand == "vselr")) {
        if (failed(applyKeepLiveExpandAnnotate(module, expand, diag)))
          return failure();
      }
    } else if (plan.materializeBoundary || expand == "e2b" || expand == "brc") {
      if (expand != "e2b" && expand != "brc") {
        diag << "layout plan: MaterializeBoundary requires Expand(e2b|brc)\n";
        return failure();
      }
      if (failed(applyMaterializeExpand(module, expand, diag)))
        return failure();
    }
  }

  VMILayoutPlan stamped = plan;
  stamped.preferredExpand = expand;
  if (stamped.source.empty())
    stamped.source = "auto";
  setLayoutPlanAttr(module, stamped);

  if (!expand.empty())
    module->setAttr("pto.vmi.preferred_expand",
                    StringAttr::get(module.getContext(), expand));
  return success();
}

void mlir::pto::setLayoutPlanAttr(ModuleOp module, const VMILayoutPlan &plan) {
  module->setAttr(layoutPlanAttrName(),
                  encodeLayoutPlanAttr(module.getContext(), plan));
}

std::optional<VMILayoutPlan> mlir::pto::getLayoutPlanAttr(ModuleOp module) {
  auto attr = module->getAttrOfType<DictionaryAttr>(layoutPlanAttrName());
  if (!attr)
    return std::nullopt;
  std::string unused;
  llvm::raw_string_ostream diag(unused);
  FailureOr<VMILayoutPlan> parsed = parseLayoutPlanAttr(attr, diag);
  if (failed(parsed))
    return std::nullopt;
  return *parsed;
}
