else if (op == "Abs") {
  return import_op_Abs(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Acos") {
  return import_op_Acos(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Acosh") {
  return import_op_Acosh(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Add") {
  return import_op_Add(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "AffineGrid") {
  return import_op_AffineGrid(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "And") {
  return import_op_And(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ArgMax") {
  return import_op_ArgMax(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Asin") {
  return import_op_Asin(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Asinh") {
  return import_op_Asinh(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Atan") {
  return import_op_Atan(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Atanh") {
  return import_op_Atanh(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Attention") {
  return import_op_Attention(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "AveragePool") {
  return import_op_AveragePool(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "BatchNormalization") {
  return import_op_BatchNormalization(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Bernoulli") {
  return import_op_Bernoulli(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "BitShift") {
  return import_op_BitShift(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "BitwiseAnd") {
  return import_op_BitwiseAnd(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "BitwiseNot") {
  return import_op_BitwiseNot(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "BitwiseOr") {
  return import_op_BitwiseOr(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "BitwiseXor") {
  return import_op_BitwiseXor(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "BlackmanWindow") {
  return import_op_BlackmanWindow(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Cast") {
  return import_op_Cast(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "CastLike") {
  return import_op_CastLike(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Ceil") {
  return import_op_Ceil(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Celu") {
  return import_op_Celu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "CenterCropPad") {
  return import_op_CenterCropPad(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Clip") {
  return import_op_Clip(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Col2Im") {
  return import_op_Col2Im(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Compress") {
  return import_op_Compress(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Concat") {
  return import_op_Concat(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ConcatFromSequence") {
  return import_op_ConcatFromSequence(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Constant") {
  return import_op_Constant(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ConstantOfShape") {
  return import_op_ConstantOfShape(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Conv") {
  return import_op_Conv(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ConvInteger") {
  return import_op_ConvInteger(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ConvTranspose") {
  return import_op_ConvTranspose(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Cos") {
  return import_op_Cos(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Cosh") {
  return import_op_Cosh(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "CumSum") {
  return import_op_CumSum(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "DFT") {
  return import_op_DFT(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "DeformConv") {
  return import_op_DeformConv(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "DepthToSpace") {
  return import_op_DepthToSpace(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "DequantizeLinear") {
  return import_op_DequantizeLinear(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Det") {
  return import_op_Det(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Div") {
  return import_op_Div(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Dropout") {
  return import_op_Dropout(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "DynamicQuantizeLinear") {
  return import_op_DynamicQuantizeLinear(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Einsum") {
  return import_op_Einsum(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Elu") {
  return import_op_Elu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Equal") {
  return import_op_Equal(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Erf") {
  return import_op_Erf(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Exp") {
  return import_op_Exp(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Expand") {
  return import_op_Expand(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "EyeLike") {
  return import_op_EyeLike(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Flatten") {
  return import_op_Flatten(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Floor") {
  return import_op_Floor(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GRU") {
  return import_op_GRU(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Gather") {
  return import_op_Gather(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GatherElements") {
  return import_op_GatherElements(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GatherND") {
  return import_op_GatherND(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Gelu") {
  return import_op_Gelu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Gemm") {
  return import_op_Gemm(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GlobalAveragePool") {
  return import_op_GlobalAveragePool(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GlobalLpPool") {
  return import_op_GlobalLpPool(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GlobalMaxPool") {
  return import_op_GlobalMaxPool(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Greater") {
  return import_op_Greater(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GreaterOrEqual") {
  return import_op_GreaterOrEqual(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GridSample") {
  return import_op_GridSample(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "GroupNormalization") {
  return import_op_GroupNormalization(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "HammingWindow") {
  return import_op_HammingWindow(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "HannWindow") {
  return import_op_HannWindow(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "HardSigmoid") {
  return import_op_HardSigmoid(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "HardSwish") {
  return import_op_HardSwish(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Hardmax") {
  return import_op_Hardmax(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Identity") {
  return import_op_Identity(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "If") {
  return import_op_If(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ImageDecoder") {
  return import_op_ImageDecoder(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "InstanceNormalization") {
  return import_op_InstanceNormalization(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "IsInf") {
  return import_op_IsInf(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "IsNaN") {
  return import_op_IsNaN(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "LRN") {
  return import_op_LRN(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "LSTM") {
  return import_op_LSTM(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "LayerNormalization") {
  return import_op_LayerNormalization(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "LeakyRelu") {
  return import_op_LeakyRelu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Less") {
  return import_op_Less(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "LessOrEqual") {
  return import_op_LessOrEqual(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Log") {
  return import_op_Log(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "LogSoftmax") {
  return import_op_LogSoftmax(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Loop") {
  return import_op_Loop(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "LpNormalization") {
  return import_op_LpNormalization(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "LpPool") {
  return import_op_LpPool(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "MatMul") {
  return import_op_MatMul(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "MatMulInteger") {
  return import_op_MatMulInteger(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Max") {
  return import_op_Max(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "MaxPool") {
  return import_op_MaxPool(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "MaxRoiPool") {
  return import_op_MaxRoiPool(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "MaxUnpool") {
  return import_op_MaxUnpool(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Mean") {
  return import_op_Mean(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "MeanVarianceNormalization") {
  return import_op_MeanVarianceNormalization(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "MelWeightMatrix") {
  return import_op_MelWeightMatrix(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Min") {
  return import_op_Min(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Mish") {
  return import_op_Mish(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Mod") {
  return import_op_Mod(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Mul") {
  return import_op_Mul(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Multinomial") {
  return import_op_Multinomial(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Neg") {
  return import_op_Neg(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "NegativeLogLikelihoodLoss") {
  return import_op_NegativeLogLikelihoodLoss(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "NonMaxSuppression") {
  return import_op_NonMaxSuppression(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "NonZero") {
  return import_op_NonZero(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Not") {
  return import_op_Not(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "OneHot") {
  return import_op_OneHot(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Optional") {
  return import_op_Optional(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "OptionalGetElement") {
  return import_op_OptionalGetElement(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "OptionalHasElement") {
  return import_op_OptionalHasElement(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Or") {
  return import_op_Or(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "PRelu") {
  return import_op_PRelu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Pad") {
  return import_op_Pad(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Pow") {
  return import_op_Pow(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "QLinearConv") {
  return import_op_QLinearConv(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "QLinearMatMul") {
  return import_op_QLinearMatMul(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "QuantizeLinear") {
  return import_op_QuantizeLinear(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RMSNormalization") {
  return import_op_RMSNormalization(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RNN") {
  return import_op_RNN(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RandomNormal") {
  return import_op_RandomNormal(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RandomNormalLike") {
  return import_op_RandomNormalLike(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RandomUniform") {
  return import_op_RandomUniform(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RandomUniformLike") {
  return import_op_RandomUniformLike(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Range") {
  return import_op_Range(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Reciprocal") {
  return import_op_Reciprocal(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceL1") {
  return import_op_ReduceL1(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceL2") {
  return import_op_ReduceL2(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceLogSum") {
  return import_op_ReduceLogSum(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceLogSumExp") {
  return import_op_ReduceLogSumExp(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceMax") {
  return import_op_ReduceMax(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceMean") {
  return import_op_ReduceMean(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceMin") {
  return import_op_ReduceMin(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceProd") {
  return import_op_ReduceProd(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceSum") {
  return import_op_ReduceSum(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReduceSumSquared") {
  return import_op_ReduceSumSquared(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RegexFullMatch") {
  return import_op_RegexFullMatch(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Relu") {
  return import_op_Relu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Reshape") {
  return import_op_Reshape(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ReverseSequence") {
  return import_op_ReverseSequence(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RoiAlign") {
  return import_op_RoiAlign(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "RotaryEmbeeding") {
  return import_op_RotaryEmbeeding(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Round") {
  return import_op_Round(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "STFT") {
  return import_op_STFT(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Scan") {
  return import_op_Scan(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Scatter") {
  return import_op_Scatter(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ScatterElements") {
  return import_op_ScatterElements(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ScatterND") {
  return import_op_ScatterND(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Selu") {
  return import_op_Selu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SequenceAt") {
  return import_op_SequenceAt(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SequenceConstruct") {
  return import_op_SequenceConstruct(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SequenceEmpty") {
  return import_op_SequenceEmpty(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SequenceErase") {
  return import_op_SequenceErase(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SequenceInsert") {
  return import_op_SequenceInsert(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SequenceLength") {
  return import_op_SequenceLength(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SequenceMap") {
  return import_op_SequenceMap(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Shape") {
  return import_op_Shape(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Shrink") {
  return import_op_Shrink(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Sigmoid") {
  return import_op_Sigmoid(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Sign") {
  return import_op_Sign(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Sin") {
  return import_op_Sin(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Sinh") {
  return import_op_Sinh(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Size") {
  return import_op_Size(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Slice") {
  return import_op_Slice(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Softmax") {
  return import_op_Softmax(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SoftmaxCrossEntropyLoss") {
  return import_op_SoftmaxCrossEntropyLoss(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Softplus") {
  return import_op_Softplus(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Softsign") {
  return import_op_Softsign(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SpaceToDepth") {
  return import_op_SpaceToDepth(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Split") {
  return import_op_Split(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "SplitToSequence") {
  return import_op_SplitToSequence(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Sqrt") {
  return import_op_Sqrt(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Squeeze") {
  return import_op_Squeeze(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "StringConcat") {
  return import_op_StringConcat(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "StringNormalizer") {
  return import_op_StringNormalizer(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "StringSplit") {
  return import_op_StringSplit(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Sub") {
  return import_op_Sub(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Sum") {
  return import_op_Sum(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Swish") {
  return import_op_Swish(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Tan") {
  return import_op_Tan(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Tanh") {
  return import_op_Tanh(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "TensorScatter") {
  return import_op_TensorScatter(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "TfIdfVectorizer") {
  return import_op_TfIdfVectorizer(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "ThresholdedRelu") {
  return import_op_ThresholdedRelu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Tile") {
  return import_op_Tile(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "TopK") {
  return import_op_TopK(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Transpose") {
  return import_op_Transpose(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Trilu") {
  return import_op_Trilu(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Unique") {
  return import_op_Unique(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Unsqueeze") {
  return import_op_Unsqueeze(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Upsample") {
  return import_op_Upsample(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Where") {
  return import_op_Where(state, inputs, outputCount, attributes, opversion, node);
}
else if (op == "Xor") {
  return import_op_Xor(state, inputs, outputCount, attributes, opversion, node);
}
