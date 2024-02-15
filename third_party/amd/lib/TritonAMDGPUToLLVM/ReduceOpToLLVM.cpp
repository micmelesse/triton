#include "ReduceOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::linearize;
using ::mlir::LLVM::AMD::loadShared;
using ::mlir::LLVM::AMD::shflSync;
using ::mlir::LLVM::AMD::storeShared;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::AMD::TritonGPUToLLVMTypeConverter;
using ::AMD::ConvertTritonGPUOpToLLVMPatternBase;
using ::AMD::ConvertTritonGPUOpToLLVMPattern;


static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}
namespace AMD{
namespace {
struct ReduceOpPromotionConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpPromotionConversion(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
      int computeCapability, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(
            typeConverter, allocation, indexCacheInfo, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::cout << "ReduceOpPromotionConversion" << std::endl;
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    mod.dump();
#if 0
    auto opOperands = op->getOpOperands();
    std::cout << "promote operands: " << std::endl;
    // Promote operands and collect new operands
    SmallVector<Value> promotedOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      auto oldType = operand.get().getType().cast<RankedTensorType>();
      auto newType = oldType.cloneWith(std::nullopt, i32_ty);
      auto promotedVal = rewriter.create<mlir::arith::ExtSIOp>(
          op->getLoc(), newType, operand.get());
      promotedVal.dump();
      promotedOperands.push_back(promotedVal);
    }
#endif

#if 0
    // TODO: copy block
    // alter the combine block
    auto &oldCompineRegion = op.getCombineOp();
    for (Block &block : oldCompineRegion.getBlocks()) {
      std::cout << "old block" << std::endl;
      for (auto arg : block.getArguments()) {
        // arg.dump();
        arg.setType(i32_ty);
      }

      // for (Operation &op : block.getOperations()) {
      //   op.dump();
      // }

      Operation *reduceReturn = block.getTerminator();
      // reduceReturn->dump();
      for (OpOperand &o : reduceReturn->getOpOperands()) {
        o.get().setType(i32_ty);
      }
    }

    // new op
    auto newReduceOp = rewriter.create<triton::ReduceOp>(op.getLoc(), promotedOperands, adaptor.getAxis());
    addNamedAttrs(newReduceOp, adaptor.getAttributes());

    // attach new block
    auto &newCombineOp = newReduceOp.getCombineOp();
    rewriter.cloneRegionBefore(oldCompineRegion, newCombineOp, newCombineOp.end());

    // change uses for old op to new op
    for (size_t i = 0; i < op->getNumResults(); i++) {
      std::cout << "old block" << std::endl;
      Value newResult = newReduceOp->getResult(i);
      op->getResult(i).replaceAllUsesWith(newResult);
    }

    // replace old op with new op
    rewriter.replaceOp(op, newReduceOp);
    // rewriter.eraseOp(op);
#elif 0
    IRMapping mapping;
    // mapping.map(*(op.getOperands().begin()), newLoopResult);
    

    std::cout << "promote operands: " << std::endl;

    // promote input Values
    for (OpOperand &operand : op->getOpOperands()) {
      auto oldVal = operand.get();
      auto oldType = oldVal.getType().cast<RankedTensorType>();
      auto newType = oldType.cloneWith(std::nullopt, i32_ty);
      auto promotedVal =
          rewriter.create<mlir::arith::ExtSIOp>(op->getLoc(), newType, oldVal);
      mapping.map(oldVal, promotedVal);
    }



    Operation *newReduceOp = cloneWithInferType(rewriter, &(*op), mapping);
    auto typeInfer = dyn_cast<InferTypeOpInterface>(newReduceOp);
    if (typeInfer) {
      SmallVector<Type> newTypes;
      auto success = typeInfer.inferReturnTypes(
          newReduceOp->getContext(), newReduceOp->getLoc(),
          newReduceOp->getOperands(), newReduceOp->getAttrDictionary(),
          newReduceOp->getPropertiesStorage(), newReduceOp->getRegions(), newTypes);
      if (succeeded(success)) {
        for (size_t i = 0; i < newTypes.size(); i++)
          newReduceOp->getResult(i).setType(newTypes[i]);
      }
    }
    
    // output i32_ty
    // for (OpResult &newResult : newReduceOp->getResults()) {
    //   newResult.setType(i32_ty);
    // }

    // trunc first arg back to i16
    auto demotedValue =
          rewriter.create<mlir::arith::TruncIOp>(newReduceOp->getLoc(), i16_ty, newReduceOp->getResult(0));
    op->getResult(0).replaceAllUsesWith(demotedValue);


    // change uses for old op to new op
    // for (size_t i = 0; i < op->getNumResults(); i++) {
    //   std::cout << "results" << std::endl;
    //   Value newResult = newReduceOp->getResult(i);
    //   op->getResult(i).replaceAllUsesWith(newResult);
    // }

    // replace old op with new op
    rewriter.replaceOp(op, {demotedValue, newReduceOp->getResult(1)});

#elif 0
    // create mapping
    // IRMapping mapping;

    // std::cout << "map operands: " << std::endl;
    // for (OpOperand &oldOperand : op->getOpOperands()) {
    //   auto oldVal = oldOperand.get();
    //   auto oldType = oldVal.getType().cast<RankedTensorType>();
    //   auto newType = oldType.cloneWith(std::nullopt, i32_ty);
    //   auto newVal =
    //       rewriter.create<mlir::arith::ExtSIOp>(op->getLoc(), newType, oldVal);
    //   mapping.map(oldVal, newVal);
    // }



    std::cout << "map region: " << std::endl;
    for (Region &oldRegion : op->getRegions()) {
      std::cout << "map block: " << std::endl;
      for (Block &oldBlock : oldRegion.getBlocks()) {

        IRMapping opMapping;
        std::cout << "map child op: " << std::endl;
        for (Operation &oldChildOp : oldBlock.getOperations()) {
          
          std::cout << "map Operand: " << std::endl;
          for (OpOperand &oldChildOperand : oldChildOp->getOpOperands()) {
            auto oldVal = oldChildOperand.get();
            // newVal ?
            opMapping.map(oldVal, newVal);
          }

          auto newOp = oldChildOp.clone(opMapping)
          
         
          
        }
      }
    }


    // clone op
    // Operation *newReduceOp = op.clone(mapping);

    std::cout << "set result types: " << std::endl;
    for (opResult &oldResult : op->getResults()) {
      newReduce->getResult(i).setType(newTypes[i])
    }


    // post demote
    // auto demotedValue =
    //       rewriter.create<mlir::arith::TruncIOp>(newReduceOp->getLoc(), i16_ty, newReduceOp->getResult(0));
    // op->getResult(0).replaceAllUsesWith(demotedValue);

    // // replace old op with new op
    // rewriter.replaceOp(op, {demotedValue, newReduceOp->getResult(1)});
#elif 1
    std::cout << "promote operands: " << std::endl;
    SmallVector<Value> promotedOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      auto oldType = operand.get().getType().cast<RankedTensorType>();
      auto newType = oldType.cloneWith(std::nullopt, i32_ty);
      auto promotedVal = rewriter.create<mlir::arith::ExtSIOp>(
          op->getLoc(), newType, operand.get());
      promotedVal.dump();
      promotedOperands.push_back(promotedVal);
    }
    
    std::cout << "new reduce op:" << std::endl;
    // new op
    auto newReduceOp = rewriter.create<triton::ReduceOp>(op.getLoc(), promotedOperands, adaptor.getAxis());
    auto &newCompineRegion = newReduceOp.getCombineOp();
    Block* newBlock = rewriter.createBlock(&newCompineRegion);
  

    std::cout << "write new op:" << std::endl;
    // write new Block
    rewriter.setInsertionPointToStart(newBlock);
    for (Block &oldBlock : op.getCombineOp().getBlocks()) {
      std::cout << "set args" << std::endl;
      for (size_t i = 0; i < oldBlock.getNumArguments(); i++) {
        std::cout << "new arg" << std::endl;
        newBlock->addArgument(i32_ty, newReduceOp.getLoc());
      }

      std::cout << "copy ops" << std::endl;
      for (Operation &oldOp : oldBlock.getOperations()) {
        Operation *newOp = rewriter.clone(oldOp);
        // update operands
        for (OpOperand &operand : newOp->getOpOperands()) {
            auto val = operand.get();
            auto type = val.getType();
            if (type.isInteger(16)){
                val.setType(i32_ty);
            }
        }
      }
    }
    rewriter.setInsertionPointToEnd(newBlock);


#if 0
    std::cout << "trunc result: " << std::endl;
    auto demotedValue =
          rewriter.create<mlir::arith::TruncIOp>(newReduceOp->getLoc(), i16_ty, newReduceOp->getResult(0));
    op->getResult(0).replaceAllUsesWith(demotedValue);

    std::cout << "replace op: " << std::endl;
    rewriter.replaceOp(op, {demotedValue, newReduceOp->getResult(1)});
#else
    // replace uses
    for (size_t i = 0; i < op->getNumResults(); i++) {
      op->getResult(i).replaceAllUsesWith(newReduceOp->getResult(i));
    }
    rewriter.replaceOp(op, newReduceOp);
#endif
#elif 0
    // clone combine region
    // Region &oldCombineRegion = op.getCombineOp();
    // Block &oldCombineBlock = oldCombineRegion.front();

    auto newReduceOp = rewriter.create<triton::ReduceOp>(op.getLoc(), promotedOperands, adaptor.getAxis());

    // Region &newCombineRegion = newReduceOp.getCombineOp();
    // rewriter.cloneRegionBefore(oldCombineRegion, newCombineRegion,
    //                            newCombineRegion.end());


    // Block &newCombineBlock = newCombineRegion.front();
    // Operation *newReduceReturn = newCombineBlock.getTerminator();

    // mod.dump();

    // rewriter.replaceOp(op, newReduceOp.getResults());
    rewriter.eraseOp(op);
#endif 


    return success();
  }

private:
  int computeCapability;
};

struct ReduceOpConversion
    : public ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(
      TritonGPUToLLVMTypeConverter &typeConverter, ModuleAllocation &allocation,
      ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
      int computeCapability, PatternBenefit benefit)
      : ConvertTritonGPUReduceScanToLLVMPattern<triton::ReduceOp>(
            typeConverter, allocation, indexCacheInfo, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
  std::cout << "ReduceOpConversion"<< std::endl;
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  mod.dump();
#if 1
    ReduceOpHelper helper(op);
    assert(helper.isSupportedLayout() &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = op->getLoc();

    auto srcValues = unpackInputs(loc, op, adaptor, rewriter);
#elif 0

  ReduceOpHelper helper(op);
  assert(helper.isSupportedLayout() &&
         "Unexpected srcLayout in ReduceOpConversion");
  Location loc = op->getLoc();

  SmallVector<Value> promotedOperands;
  for (OpOperand &operand : op->getOpOperands()) {
    auto oldType = operand.get().getType().cast<RankedTensorType>();
    auto newType = oldType.cloneWith(std::nullopt, i32_ty);
    auto promotedVal = rewriter.create<mlir::arith::ExtSIOp>(
        op->getLoc(), newType, operand.get());
    promotedOperands.push_back(promotedVal);
  }

  llvm::SmallVector<mlir::RankedTensorType> inputTypes = op.getInputTypes();
  SmallVector<Type> elemTypes = op.getElementTypes();
  // auto operands = adaptor.getOperands();
  unsigned srcElems = getTotalElemsPerThread(inputTypes[0]);
  std::cout << "srcElems: " << srcElems << std::endl;
  SmallVector<SmallVector<Value>> srcValues(srcElems);
  for (unsigned i = 0; i < op.getNumOperands(); ++i) {
    auto llvmStruct = promotedOperands[i];
    // auto values =
    //     getTypeConverter()->unpackLLElements(loc, llvmStruct, rewriter);
    SmallVector<Value> values;
    if (llvmStruct.getType().isIntOrIndexOrFloat() ||
        llvmStruct.getType().isa<triton::PointerType>() ||
        llvmStruct.getType().isa<LLVM::LLVMPointerType>()) {
      values = {llvmStruct};
    } else {
      ArrayRef<Type> llvmTypes =
          llvmStruct.getType().cast<LLVM::LLVMStructType>().getBody();
      
      SmallVector<Value> results(llvmTypes.size());
      for (unsigned j = 0; j < llvmTypes.size(); ++j) {
        Type type = llvmTypes[i];
        results[i] = extract_val(type, llvmStruct, j);
      }
      values = results;
    }

    assert(values.size() == srcValues.size());
    for (unsigned j = 0; j < srcValues.size(); ++j) {
      srcValues[j].push_back(values[j]);
    }
  }

#elif 0
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      // unpack the operands
      SmallVector<Value> values;

      auto operand = operands[i];
      if (operand.getType().isIntOrIndexOrFloat() ||
          operand.getType().isa<triton::PointerType>() ||
          operand.getType().isa<LLVM::LLVMPointerType>()) {
        values = {operand};
      } else {
        ArrayRef<Type> types =
            operand.getType().cast<LLVM::LLVMStructType>().getBody();
        values.reserve(types.size());
        for (unsigned i = 0; i < types.size(); ++i) {
          Type type = types[i];
          unsigned bitwidth = type.getIntOrFloatBitWidth();
          std::cout << "bitwidth: " << bitwidth << std::endl;

          Value val = extract_val(type, operand, i);
          if (bitwidth < 32) {
            std::cout << "promoting i" << bitwidth << " to i32" << std::endl;
            mod.dump();
            Value new_val = sext(i32_ty, val);
            values.push_back(new_val);
            mod.dump();
            val.dump();
          }else{
            values.push_back(val);
          }

             
        }
      }

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
#elif 0
    mlir::ValueRange operands =
        adaptor.getOperands(); // iterator over values of type (tensor<128xi16>,
                               // tensor<128xi32>)
    unsigned numOperands = op.getNumOperands(); // 2
    llvm::SmallVector<mlir::RankedTensorType> types =
        op.getInputTypes(); // (tensor<128xi16>, tensor<128xi32>)
    llvm::SmallVector<mlir::Type> elemTypes = op.getElementTypes(); // i16, i32

    // NOTE: just operate on operands[0] tensor<128xi16>
    mlir::Value operand = operands[0];           // value of tensor<128xi16>
    mlir::Type operand_type = operand.getType(); // tensor<128xi16>

    if (isa<LLVM::LLVMStructType>(operand_type)) {
      std::cout << "operand is an LLVMStructType" << std::endl;
      operand_type.dump();
      mlir::Type operand_element_type = elemTypes[0]; // i16
      if (operand_element_type.isInteger(16)) {
        std::cout << "operand element type is i16" << std::endl;

        // just use second arg which is tensor<128xi32>
        SmallVector<mlir::Value> newOperands(numOperands);
        newOperands[0] = operands[1];
        newOperands[1] = operands[1];
        llvm::SmallVector<mlir::Type> newTypes(numOperands);
        newTypes[0] = elemTypes[1];
        newTypes[1] = elemTypes[1];

        // new state
        OperationState newReduceState(op->getLoc(), op->getName());
        newReduceState.addOperands(newOperands);
        newReduceState.addTypes(newTypes);
        newReduceState.addAttributes(op->getAttrs());
        auto newReduce = rewriter.create(newReduceState);
        rewriter.replaceOp(op, newReduce->getResults());
        // rewriter.replaceOp(op, newReduce);

        mod.dump();
        // rewriter.replaceOpWithNewOp<triton::ReduceOp>(op, newReduceState);
        return success();
      }
    } else {
      std::cout << "operand is not an LLVMStructType" << std::endl;
      operand_type.dump();
    }
#elif 0
    // dump input module state
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    mod.dump();

    mlir::ValueRange operands = adaptor.getOperands(); // iterator over values of type (tensor<128xi16>, tensor<128xi32>)
    unsigned numOperands = op.getNumOperands(); // 2
    llvm::SmallVector<mlir::RankedTensorType> types =
        op.getInputTypes(); // (tensor<128xi16>, tensor<128xi32>)
    llvm::SmallVector<mlir::Type> elemTypes = op.getElementTypes(); // i16, i32
    unsigned axis = op.getAxis();

    // just use second arg which is tensor<128xi32>
    SmallVector<mlir::Value> newOperands(numOperands);
    newOperands[0] = operands[1];
    newOperands[1] = operands[1];
    llvm::SmallVector<mlir::Type> newTypes(numOperands);
    newTypes[0] = elemTypes[1];
    newTypes[1] = elemTypes[1];

    // new op state
    OperationState newReduceState(op->getLoc(), op->getName());
    newReduceState.addOperands(newOperands);
    newReduceState.addTypes(newTypes);
    newReduceState.addAttributes(op->getAttrs());
    newReduceState.addRegions(op->getRegions());
    Operation* newReduce = rewriter.create(newReduceState);
    // auto newReduce = rewriter.create<triton::ReduceOp>(newReduceState, newOperands, axis);
    // addNamedAttrs(newReduce, adaptor.getAttributes());

    mod.dump();

    // reduce block
    // auto &newCombineOp = newReduce->getCombineOp();
    // rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
    //                            newCombineOp.end());
    // rewriter.replaceOp(op, newReduce);

    mod.dump();
    
    // extract new op
    auto srcValues = unpackInputs(loc, newReduce, adaptor, rewriter);

    // preamble
    ReduceOpHelper helper(newReduce);
    assert(helper.isSupportedLayout() &&
           "Unexpected srcLayout in ReduceOpConversion");
    Location loc = newReduce->getLoc();
#endif

    std::map<SmallVector<unsigned>, SmallVector<Value>> accs;
    std::map<SmallVector<unsigned>, SmallVector<Value>> indices;
    // First reduce all the values along axis within each thread.
    reduceWithinThreads(helper, srcValues, accs, indices, rewriter);

    // Then reduce across threads within a warp.
    reduceWithinWarps(helper, accs, rewriter);

    if (helper.isWarpSynchronous()) {
      // If all the values to be reduced are within the same warp there is
      // nothing left to do.
      packResults(helper, accs, rewriter);
      return success();
    }

    // Compute a shared memory base per operand.
    auto smemShape = helper.getScratchConfig();

    SmallVector<Value> smemBases =
        getSmemBases(op, product<unsigned>(smemShape), rewriter);

    storeWarpReduceToSharedMemory(helper, accs, indices, smemBases, rewriter);

    sync(rewriter, loc, op);

    // The second round of shuffle reduction
    //   now the problem size: sizeInterWarps, s1, s2, .. , sn
    //   where sizeInterWarps is 2^m
    //
    // Each thread needs to process:
    //   elemsPerThread = sizeInterWarps * s1 * s2 .. Sn / numThreads
    accumulatePartialReductions(helper, smemBases, rewriter);

    // We could avoid this barrier in some of the layouts, however this is not
    // the general case.
    // TODO: optimize the barrier in case the layouts are accepted.
    sync(rewriter, loc, op);

    // set output values
    loadReductionAndPackResult(helper, smemShape, smemBases, rewriter);


    std::cout << "ReduceOpConversion Result:"<< std::endl;
    mod.dump();
    return success();
  }

private:
  int computeCapability;

  void accumulate(ConversionPatternRewriter &rewriter, Region &combineOp,
                  SmallVector<Value> &acc, ValueRange cur, bool isFirst) const {
    if (isFirst) {
      acc = SmallVector<Value>(cur.begin(), cur.end());
      return;
    }

    // Create a new copy of the reduce block, and inline it
    Block *currentBlock = rewriter.getBlock();
    Region &parent = *currentBlock->getParent();
    rewriter.cloneRegionBefore(combineOp, &parent.front());
    auto &newReduce = parent.front();
    auto returnOp = dyn_cast<triton::ReduceReturnOp>(newReduce.getTerminator());

    llvm::SmallVector<Value> combineArgs(2 * acc.size());
    for (unsigned i = 0; i < acc.size(); ++i) {
      combineArgs[i] = acc[i];
      combineArgs[acc.size() + i] = cur[i];
    }

    rewriter.inlineBlockBefore(&newReduce, &*rewriter.getInsertionPoint(),
                               combineArgs);

    auto results = returnOp.getResult();
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }

    // Delete the terminator, which is no longer used
    rewriter.eraseOp(returnOp);
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto types = op.getInputTypes();
    auto operands = adaptor.getOperands();
    unsigned srcElems = getTotalElemsPerThread(types[0]);
    SmallVector<SmallVector<Value>> srcValues(srcElems);
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto values = getTypeConverter()->unpackLLElements(loc, operands[i], rewriter);

      assert(values.size() == srcValues.size());
      for (unsigned j = 0; j < srcValues.size(); ++j) {
        srcValues[j].push_back(values[j]);
      }
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc,
            triton::ReduceOp op) const {
    // TODO[shuhaoj]: change hard code style of numThreads. Hide async_agent
    // attr.
    if (getWSAgentId(op)) {
      barSync(rewriter, op, getAgentIds(op).front(), 128);
    } else {
      barrier();
    }
  }

  // Check if the reduction can use a redux op and return the kind.
  std::optional<NVVM::ReduxKind> matchReduxKind(triton::ReduceOp op) const {
    #ifdef USE_ROCM
      return std::nullopt;
    #endif
    if (computeCapability < 80)
      return std::nullopt;
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return std::nullopt;
    Block *block = &(*op.getCombineOp().begin());
    Operation *yield = block->getTerminator();
    Operation *reduceOp = yield->getOperand(0).getDefiningOp();
    if (!reduceOp || reduceOp->getNumOperands() != 2 ||
        reduceOp->getNumResults() != 1)
      return std::nullopt;
    auto intType = reduceOp->getResultTypes()[0].dyn_cast<IntegerType>();
    if (!intType || intType.getWidth() > 32)
      return std::nullopt;
    if (reduceOp->getOperand(0) != block->getArgument(0) ||
        reduceOp->getOperand(1) != block->getArgument(1))
      return std::nullopt;
    if (isa<arith::AddIOp>(reduceOp))
      return NVVM::ReduxKind::ADD;
    if (isa<arith::AndIOp>(reduceOp))
      return NVVM::ReduxKind::AND;
    if (isa<arith::OrIOp>(reduceOp))
      return NVVM::ReduxKind::OR;
    if (isa<arith::XOrIOp>(reduceOp))
      return NVVM::ReduxKind::XOR;
    if (isa<arith::MinSIOp>(reduceOp))
      return NVVM::ReduxKind::MIN;
    if (isa<arith::MinUIOp>(reduceOp))
      return NVVM::ReduxKind::UMIN;
    if (isa<arith::MaxSIOp>(reduceOp))
      return NVVM::ReduxKind::MAX;
    if (isa<arith::MaxUIOp>(reduceOp))
      return NVVM::ReduxKind::UMAX;
    return std::nullopt;
  }

  // Reduce along op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  void reduceWithinThreads(
      ReduceOpHelper &helper, SmallVector<SmallVector<Value>> &srcValues,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    RankedTensorType operandType = op.getInputTypes()[0];
    // Assumes offsets don't actually depend on type
    SmallVector<SmallVector<unsigned>> offset =
        emitOffsetForLayout(helper.getSrcLayout(), operandType);
    unsigned srcElems = getTotalElemsPerThread(operandType);
    auto *combineOp = &op.getCombineOp();
    auto srcIndices =
        emitIndices(op.getLoc(), rewriter, helper.getSrcLayout(), operandType);
    // reduce within threads
    for (unsigned i = 0; i < srcElems; ++i) {
      SmallVector<unsigned> key = offset[i];
      key[op.getAxis()] = 0;
      bool isFirst = accs.find(key) == accs.end();
      accumulate(rewriter, *combineOp, accs[key], srcValues[i], isFirst);
      if (isFirst)
        indices[key] = srcIndices[i];
    }
  }

  // Apply warp reduction across the given number of contiguous lanes using op
  // region and the accumulator values as source.
  void warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                  SmallVector<Value> &acc, triton::ReduceOp op,
                  unsigned numLaneToReduce, unsigned interleave) const {
    if (auto kind = matchReduxKind(op)) {
      // Based on benchmarking on A100 redux op gives a speed up only when doing
      // a single reduction (not partioned) and when the mask is static.
      // Therefore we currently only enable it to reduce across all the lanes.
      if (numLaneToReduce == 32) {
        assert(acc.size() == 1);
        Value mask = i32_val(0xFFFFFFFF);
        // Even though we currently don't use redux for partitioned reduction
        // the code below supports it in case we want to tweak the heuristic.
        if (numLaneToReduce < 32) {
          // For partitioned reduction we need to caluclate the mask so that
          // each group of numLaneToReduce threads has the correct mask.
          unsigned bitmask = (1 << numLaneToReduce) - 1;
          Value threadId = getThreadId(rewriter, loc);
          Value laneId = urem(threadId, i32_val(32));
          mask = shl(i32_val(bitmask),
                     and_(laneId, i32_val(~(numLaneToReduce - 1))));
        }
        for (unsigned i = 0; i < acc.size(); ++i) {
          unsigned bitwidth = acc[i].getType().cast<IntegerType>().getWidth();
          if (bitwidth < 32) {
            if (*kind == NVVM::ReduxKind::MIN || *kind == NVVM::ReduxKind::MAX)
              acc[i] = sext(i32_ty, acc[i]);
            else
              acc[i] = zext(i32_ty, acc[i]);
          }
          acc[i] = rewriter.create<NVVM::ReduxOp>(loc, acc[i].getType(), acc[0],
                                                  *kind, mask);
          if (bitwidth < 32)
            acc[i] = trunc(int_ty(bitwidth), acc[i]);
        }
        return;
      }
    }

    for (unsigned N = numLaneToReduce / 2; N > 0; N >>= 1) {
      SmallVector<Value> shfl(acc.size());
      unsigned shuffleIdx = N;
#ifdef USE_ROCM
      auto srcTys = op.getInputTypes();
      auto inputTy = srcTys[0].cast<RankedTensorType>();
      auto inMfma =
        inputTy.getEncoding().dyn_cast<triton::gpu::MfmaEncodingAttr>();
      if (inMfma && inMfma.getIsTransposed()) {
        assert(numLaneToReduce == 2 || numLaneToReduce == 4);
        // for mfma 32x32 adjacent threads in y dimension in transposed MFMA
        // layout are 32 apart: [[0 0 0 0 32 32 32 32 ...] [1 1 1 1 33 33 33 33
        // ...] ...]. for mfma 16x16 adjacent threads in y dimension in
        // transposed MFMA layout are 16 apart: [[0 0 0 0 16 16 16 16 32 32 32
        // 32 ...] [1 1 1 1 33 33 33 33 ...] ...].
        const int warpSize = 64;
        shuffleIdx = warpSize / N / 2;
      }
#endif
      for (unsigned i = 0; i < acc.size(); ++i) {
        shfl[i] = shflSync(loc, rewriter, acc[i], shuffleIdx * interleave);
      }
      accumulate(rewriter, op.getCombineOp(), acc, shfl, false);
    }
  }

  // Reduce across threads within each warp.
  void
  reduceWithinWarps(ReduceOpHelper &helper,
                    std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                    ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    unsigned sizeIntraWarps = helper.getIntraWarpSizeWithUniqueData();
    unsigned threadOffsetOnReductionAxis =
        helper.getThreadOffsetOnReductionAxis();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = accs[key];
      warpReduce(rewriter, op.getLoc(), acc, op, sizeIntraWarps,
                 threadOffsetOnReductionAxis);
    }
  }

  // Pack the accumualtor values and replace the reduce op with the result.
  void packResults(ReduceOpHelper &helper,
                   std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    unsigned axis = op.getAxis();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        SmallVector<SmallVector<unsigned>> resultOffset =
            emitOffsetForLayout(resultLayout, resultTy);
        SmallVector<Value> resultVals;
        for (int j = 0; j < resultElems; j++) {
          auto key = resultOffset[j];
          key.insert(key.begin() + axis, 0);
          resultVals.push_back(accs[key][i]);
        }
        results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                        rewriter, resultTy);
      } else
        results[i] = accs.begin()->second[i];
    }
    rewriter.replaceOp(op, results);
  }

  SmallVector<Value>
  getMultiDimWarpId(ReduceOpHelper &helper, Value &warpId, Location &loc,
                    ConversionPatternRewriter &rewriter) const {
    auto srcLayout = helper.getSrcLayout();
    auto srcShape = helper.getSrcShape();
    auto order = getOrder(srcLayout);
    SmallVector<Value> multiDimWarpId;

    // 2x2 warps with slice dim = 0, warpId = 2 ends up writing at the same
    // address as warpId = 0 since the warpsPerCTA is [1, 2], need to figure out
    // a way to properly delinearize warpId in the slice case
    if (auto sliceLayout = srcLayout.dyn_cast<SliceEncodingAttr>()) {
      auto parentLayout = sliceLayout.getParent();
      auto parentWarpsPerCTA = triton::gpu::getWarpsPerCTA(parentLayout);
      auto parentOrder = triton::gpu::getOrder(parentLayout);
      multiDimWarpId =
          delinearize(rewriter, loc, warpId, parentWarpsPerCTA, parentOrder);
      multiDimWarpId.erase(multiDimWarpId.begin() + sliceLayout.getDim());
    } else {
      auto warpsPerCTA =
          triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape);
      multiDimWarpId = delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    }
    return multiDimWarpId;
  }

  void storeWarpReduceToSharedMemory(
      ReduceOpHelper &helper,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &accs,
      std::map<SmallVector<unsigned>, SmallVector<Value>> &indices,
      SmallVector<Value> &smemBases,
      ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    Value threadId = getThreadId(rewriter, loc);
    auto srcLayout = helper.getSrcLayout();
    unsigned wavefront_size = triton::gpu::getWarpSize(srcLayout);
    Value warpSize = i32_val(wavefront_size);
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);
    auto srcShape = helper.getSrcShape();
    unsigned axis = op.getAxis();
    auto smemShape = helper.getScratchConfig();

    auto threadsPerWarp =
        triton::gpu::getThreadsPerWarpWithUniqueData(srcLayout, srcShape);
    auto order = getOrder(srcLayout);
    SmallVector<Value> multiDimLaneId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);
    Value laneIdAxis = multiDimLaneId[axis];
    Value zero = i32_val(0);
    Value laneZero = icmp_eq(laneIdAxis, zero);

    SmallVector<Value> multiDimWarpId =
        getMultiDimWarpId(helper, warpId, loc, rewriter);
    Value warpIdAxis = multiDimWarpId[axis];

    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    for (auto it : accs) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &acc = it.second;

      SmallVector<Value> writeIdx = indices[key];
      writeIdx[axis] = warpIdAxis;
      Value writeOffset =
          linearize(rewriter, loc, writeIdx, smemShape, smemOrder);
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value writePtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                             smemBases[i], writeOffset);
        storeShared(rewriter, loc, writePtr, acc[i], laneZero);
      }
    }
  }

  // Load the reduction of each warp and accumulate them to a final value and
  // store back to shared memory.
  void accumulatePartialReductions(ReduceOpHelper &helper,
                                   SmallVector<Value> &smemBases,
                                   ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    auto srcLayout = helper.getSrcLayout();
    auto smemShape = helper.getScratchConfig();
    unsigned elems = product<unsigned>(smemShape);
    unsigned sizeInterWarps = helper.getInterWarpSizeWithUniqueData();
    Location loc = op.getLoc();

    Value threadId = getThreadId(rewriter, loc);
    unsigned wavefront_size = triton::gpu::getWarpSize(srcLayout);
    Value warpSize = i32_val(wavefront_size);
    Value laneId = urem(threadId, warpSize);
    Value zero = i32_val(0);

    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    unsigned numThreads =
        product<unsigned>(triton::gpu::getWarpsPerCTA(srcLayout)) *
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned elemsPerThread = std::max<unsigned>(elems / numThreads, 1);
    Value threadIsNeeded = icmp_slt(threadId, i32_val(elems));
    Value readOffset = threadId;
    for (unsigned round = 0; round < elemsPerThread; ++round) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        Value readPtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                            smemBases[i], readOffset);
        acc[i] = loadShared(rewriter, loc, readPtr, elemTy, threadIsNeeded);
      }
      warpReduce(rewriter, loc, acc, op, sizeInterWarps, 1 /* interleave */);
      // only the first thread in each sizeInterWarps is writing
      Value writeOffset = readOffset;
      SmallVector<Value> writePtrs(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto elemTy = getElementType(op, i);
        writePtrs[i] = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                           smemBases[i], writeOffset);
      }

      Value laneIdModSizeInterWarps = urem(laneId, i32_val(sizeInterWarps));
      Value laneIdModSizeInterWarpsIsZero =
          icmp_eq(laneIdModSizeInterWarps, zero);
      Value pred = and_(threadIsNeeded, laneIdModSizeInterWarpsIsZero);
      unsigned wavefront_size = triton::gpu::getWarpSize(srcLayout);

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
#if USE_ROCM
        // This barrier is known to be critical for Navi 2x/3x
        if (i > 0 && wavefront_size == 32) {
            GCNBuilder BuilderMemfenceLDS;
            BuilderMemfenceLDS.create<>("s_waitcnt lgkmcnt(0)")->operator()();
            BuilderMemfenceLDS.launch(rewriter, loc, void_ty(rewriter.getContext()));
        }
#endif
        storeShared(rewriter, loc, writePtrs[i], acc[i], pred);
      }

      if (round != elemsPerThread - 1) {
        readOffset = add(readOffset, i32_val(numThreads));
      }
    }
  }

  // Load the final reduction from shared memory and replace the reduce result
  // with it.
  void loadReductionAndPackResult(ReduceOpHelper &helper,
                                  SmallVector<unsigned> smemShape,
                                  SmallVector<Value> &smemBases,
                                  ConversionPatternRewriter &rewriter) const {
    triton::ReduceOp op = helper.getOperation();
    Location loc = op.getLoc();
    auto srcLayout = helper.getSrcLayout();
    auto axis = op.getAxis();
    auto smemOrder = helper.getOrderWithAxisAtBeginning();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      auto elemTy = getElementType(op, i);
      if (auto resultTy =
              op.getResult()[i].getType().dyn_cast<RankedTensorType>()) {
        // nd-tensor where n >= 1
        auto resultLayout = resultTy.getEncoding().cast<SliceEncodingAttr>();
        unsigned resultElems = getTotalElemsPerThread(resultTy);
        auto resultIndices = emitIndices(loc, rewriter, resultLayout, resultTy);
        assert(resultIndices.size() == resultElems);

        SmallVector<Value> resultVals(resultElems);
        for (size_t j = 0; j < resultElems; ++j) {
          SmallVector<Value> readIdx = resultIndices[j];
          readIdx.insert(readIdx.begin() + op.getAxis(), i32_val(0));
          Value readOffset =
              linearize(rewriter, loc, readIdx, smemShape, smemOrder);
          Value readPtr = gep(ptr_ty(rewriter.getContext(), 3), elemTy,
                              smemBases[i], readOffset);
          resultVals[j] = load(elemTy, readPtr);
        }

        results[i] = getTypeConverter()->packLLElements(loc, resultVals,
                                                        rewriter, resultTy);
      } else {
        // 0d-tensor -> scalar
        results[i] = load(elemTy, smemBases[i]);
      }
    }
    rewriter.replaceOp(op, results);
  }
};
}

void populateReduceOpToLLVMPatterns(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int numWarps, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    ModuleAllocation &allocation,
    ConvertTritonGPUOpToLLVMPatternBase::IndexCacheInfo &indexCacheInfo,
    int computeCapability, PatternBenefit benefit) {

#if 1
  patterns.add<ReduceOpPromotionConversion>(
      typeConverter, allocation, indexCacheInfo, computeCapability, 2);
  patterns.add<ReduceOpConversion>(typeConverter, allocation, indexCacheInfo,
                                   computeCapability, 1);
#elif 0
  patterns.add<ReduceOpConversion>(typeConverter, allocation, indexCacheInfo,
                                   computeCapability, benefit);
#endif
}
}
