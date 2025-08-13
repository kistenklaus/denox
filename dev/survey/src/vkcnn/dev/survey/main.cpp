

#include "src/vkcnn/dev/survey/conv.hpp"
#include "vkcnn/common/ActivationFunction.hpp"
#include "vkcnn/dev/utils/merian.hpp"
#include "vkcnn/shaders/conv/ConvTemplate.hpp"

vkcnn::dev::survey::ConvSurvey
survey_conv(const merian::ContextHandle &context) {

  // vkcnn::shaders::Conv3x3mma16x8x8_CHWC8_RCSKC8_HR_P2
  //     mma16x8x8_chwc8_rcskc8_hr_p2;
  // vkcnn::shaders::Conv3x3mmaVectorized mma16x8x8_vectorized{
  //     glm::uvec3(16, 8, 8)};
  // vkcnn::shaders::Conv3x3mmaVectorized mma16x16x16_vectorized{
  //     glm::uvec3(16, 16, 16)};

  std::vector<vkcnn::shaders::ConvTemplate *> shaders = {};
  using namespace vkcnn;
  using namespace vkcnn::dev::survey;

  ConvMatrix config = {
      .inout =
          {
              {1920, 1080, 3, 3},   //
              {1920, 1080, 3, 9},   //
              {1920, 1080, 9, 32},  //
              {1920, 1080, 32, 32}, //
              {1920, 1080, 73, 64}, //
              {1920, 1080, 64, 64}, //
              {1920, 1080, 64, 32}, //
              {1920, 1080, 32, 3},  //

              {960, 540, 32, 48},  //
              {960, 540, 48, 48},  //
              {960, 540, 128, 64}, //
              {960, 540, 64, 64},  //

              {480, 270, 64, 64},  //
              {480, 270, 160, 96}, //
              {480, 270, 96, 96},  //

              {240, 135, 80, 80},   //
              {240, 135, 144, 112}, //
              {240, 135, 112, 112}, //

              {120, 68, 96, 96}, //
          },
      .kernelSize =
          {
              glm::uvec2(3, 3) //
          },
      .types =
          {
              {FloatType::F16, FloatType::F16, FloatType::F16}, //
          },
      .layouts =
          {
              {ActivationLayout::CHWC8, ActivationLayout::CHWC8},
              {ActivationLayout::CHWC16, ActivationLayout::CHWC16},
          },
      .padStride = {ConvPadStride(glm::uvec2(3, 3), glm::uvec2(1, 1))},
      .activationFunctions = {
          std::nullopt, //
      }};

  return vkcnn::dev::survey::conv(context, shaders, config);
}

int main() {
  merian::ContextHandle context = vkcnn::merian::createContext();
  {
    auto survey = survey_conv(context);
    survey.print();
  }
}
