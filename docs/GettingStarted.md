1. **Preparing the model** <br>
   *denox* accepts ONNX models as the network description.
   Trained pytorch or tensorflow models can be exported to 
   ONNX.
   For example with pytorch
    ```Python
    net = Net() # <- your pytorch model!
    example_input = torch.ones(1, INPUT_CHANNELS_COUNT, 5, 5, dtype=torch.float16)
    program = torch.onnx.export(
        net,
        (example_input,),
        dynamic_shapes={
            "input": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC}
        },
        input_names=["input"],
        output_names=["output"],
    )
    program.save("net.onnx")
    ```
    We advise giving inputs and outputs names, as we will later 
    need those to refer to specific tensors.
    Additionally above, we compile the model with a dynamic input size,
    if this is not specified the model will only work for the 
    shape of the `example_input`.
    Further if dynamic shapes are used *denox* expects models to 
    be valid for all input sizes. 
    For U-Net architectures with skip-connections this means that 
    inputs must be explicitly aligned.
    ```Python
    class UNetAlignment(nn.Module):
        def __init__(self, net):
            super(UNetAlignment, self).__init__()
            self.net = net

        def forward(self, input):
            alignment = self.net.alignment
            H, W = input.size(2), input.size(3)
            pad_w = (alignment - (W % alignment)) % alignment
            pad_h = (alignment - (H % alignment)) % alignment
            aligned = F.pad(input, (0, pad_w, 0, pad_h), mode="replicate")
            output = self.net(aligned)
            return output[:,:,:H,:W]
    net = UNetAlignment(Net())
    ```
2. **Compile**
    ```
    denox compile net.onnx \
        --type input=f16 output=f16 \
        --storage input=ssbo output=ssbo \
        --format input=hwc output=hwc \
        --shape input=H:W:C \
        --assume H=1080 W=1920 \
        --db /tmp/gpu.db \
        -o net.dnx
    ```
    This produces a compiled model artifact `net.dnx`
    > See `denox compile --help` for more information.
3. **Benchmarking \& Testing**<br>
    - To get quick profiling results, about inference latency 
      and dispatch schedules.
      ```
      denox bench net.dnx --spec H=1080 W=1920
      ```
    - To test if compiled model contains the correct weights, 
      the easiest way is to run a single inference on a PNG. 
      ```
      denox infer net.dnx -i input.png -o output.png
      ```
4. **Integration within Renderers** <br>
    The produced artefact `net.dnx` is a flatbuffer,
    which contains a list of dispatches, including 
    SPIR-V binaries, descriptor-set bindings, 
    push-constants and dispatch sizes.
    When executed sequentially those dispatches 
    infer the model. *denox* intentionally 
    does not provide a runtime instead it relies on
    engine developers to write this parsing step manually
    either within a asset pipeline at compiletime or 
    at runtime. 
      
