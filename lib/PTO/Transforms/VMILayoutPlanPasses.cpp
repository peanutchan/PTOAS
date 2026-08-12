// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutPlanPasses.cpp - Infer/apply layout plan passes -----------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutPlan.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIINFERLAYOUTPLAN
#define GEN_PASS_DEF_VMIAPPLYLAYOUTPLAN
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct VMIInferLayoutPlanPass
    : public mlir::pto::impl::VMIInferLayoutPlanBase<VMIInferLayoutPlanPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIInferLayoutPlanPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Do not overwrite an explicit override plan already stamped on the module.
    if (auto existing = getLayoutPlanAttr(module)) {
      if (existing->source == "override")
        return;
    }
    FailureOr<VMILayoutPlan> plan = inferLayoutPlan(module, llvm::errs());
    if (failed(plan)) {
      signalPassFailure();
      return;
    }
    setLayoutPlanAttr(module, *plan);
  }
};

struct VMIApplyLayoutPlanPass
    : public mlir::pto::impl::VMIApplyLayoutPlanBase<VMIApplyLayoutPlanPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIApplyLayoutPlanPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::optional<VMILayoutPlan> plan = getLayoutPlanAttr(module);
    if (!plan) {
      module.emitError() << "vmi-apply-layout-plan requires "
                         << layoutPlanAttrName()
                         << " (run vmi-infer-layout-plan or provide an override)";
      signalPassFailure();
      return;
    }
    if (failed(applyLayoutPlan(module, *plan, llvm::errs()))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIInferLayoutPlanPass() {
  return std::make_unique<VMIInferLayoutPlanPass>();
}

std::unique_ptr<Pass> mlir::pto::createVMIApplyLayoutPlanPass() {
  return std::make_unique<VMIApplyLayoutPlanPass>();
}
