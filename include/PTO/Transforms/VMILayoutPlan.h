// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutPlan.h - First-class VMI layout plan contract -*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTPLAN_H
#define PTO_TRANSFORMS_VMILAYOUTPLAN_H

#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir::pto {

/// Residency policy override (mutually exclusive).
/// Empty string means "derive from morphs / legacy flags".
enum class VMIResidencyKind {
  Unspecified = 0,
  KeepLive,
  Materialize,
  Split,
};

/// Structured layout-algebra plan stamped on ModuleOp as `pto.vmi.layout_plan`.
struct VMILayoutPlan {
  int64_t version = 1;
  std::string source;          // "auto" | "override"
  std::string preferredExpand; // e2b|brc|vbrc|vselr|""
  int64_t group = 0;
  bool materializeBoundary = false;
  bool stageSplit = false;

  /// Explicit residency override: "keeplive" | "materialize" | "split" | "".
  std::string residency;
  /// Materialize / split placement: "vf" | "inner" | "outer" | "stage" | "".
  std::string materializeScope;
  /// Split axis name (e.g. "i_mhc"); empty when residency != split.
  std::string splitAxis;
  /// Split factor B; 0 when residency != split.
  int64_t splitFactor = 0;

  SmallVector<std::string, 8> morphs;

  bool empty() const {
    return morphs.empty() && preferredExpand.empty() && residency.empty();
  }

  VMIResidencyKind residencyKind() const;
};

StringRef layoutPlanAttrName();

FailureOr<VMILayoutPlan> parseLayoutPlanText(StringRef text,
                                             raw_ostream &diag);

FailureOr<VMILayoutPlan> parseLayoutPlanAttr(DictionaryAttr attr,
                                             raw_ostream &diag);

DictionaryAttr encodeLayoutPlanAttr(MLIRContext *ctx, const VMILayoutPlan &plan);

std::string formatLayoutPlanText(const VMILayoutPlan &plan);

/// Normalize derived fields (residency/scope/split) from morphs + legacy flags.
void normalizeLayoutPlan(VMILayoutPlan &plan);

FailureOr<VMILayoutPlan> inferLayoutPlan(ModuleOp module, raw_ostream &diag);

LogicalResult validateLayoutPlan(ModuleOp module, const VMILayoutPlan &plan,
                                 raw_ostream &diag);

LogicalResult applyLayoutPlan(ModuleOp module, const VMILayoutPlan &plan,
                              raw_ostream &diag);

void setLayoutPlanAttr(ModuleOp module, const VMILayoutPlan &plan);

std::optional<VMILayoutPlan> getLayoutPlanAttr(ModuleOp module);

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTPLAN_H
