#include <catch2/catch.hpp>
#include <tensor/nn/layer/feed_forward.hpp>
#include <tensor/nn/layer/conv_2d.hpp>
#include <tensor/nn/saver.hpp>

class Model : public ts::ParameterRegistry<float> {
  public:
    Model()
        : _layer1(ts::FeedForward::create(2, 100, ts::Activation::RELU)),
          _layer2(ts::Conv2D::create(3, 64, 3, 1, 0, 1, ts::Activation::NONE)),
          _number({1})
    {
        _number.set_weight(std::make_unique<ts::VectorF>(ts::VectorF{1337}));

       register_parameters(_layer1.parameters());
       register_parameters(_layer2.parameters());
       register_parameters(_number);
    }

  private:
    ts::FeedForward _layer1;
    ts::Conv2D _layer2;
    ts::Variable<float, 1> _number;
};

TEST_CASE("saver")
{
    {
        auto model = Model();
        auto saver = ts::Saver(model);
        saver.save("model.ts");
    }
    {
        auto model = Model();
        auto saver = ts::Saver(model);
        saver.load("model.ts");
    }
}