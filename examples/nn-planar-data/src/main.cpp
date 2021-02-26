
#include <tensor/nn/cross_entropy_loss.hpp>
#include <tensor/nn/feed_forward.hpp>
#include <tensor/nn/planar_dataset.hpp>
#include <tensor/nn/activations.hpp>

class Model {

  public:
    Model()
    : _layer1(2, 100, ts::Activation::RELU),
      _layer2(100, 3) {}

    auto predict(ts::MatrixF const &inputs) -> ts::VectorI { return ts::argmax(_forward(inputs)); }

    auto update(ts::MatrixF const &inputs, ts::VectorI const &labels, float learning_rate) -> float
    {
        // forward pass
        auto logits = _forward(inputs);
        float loss = _loss(logits, labels);

        // backward pass
        _layer1.backward(_layer2.backward(_loss.backward()));

        // parameters update
        _layer1.update(learning_rate);
        _layer2.update(learning_rate);

        return loss;
    }

  private:
    ts::FeedForward _layer1;
    ts::FeedForward _layer2;
    ts::CrossEntropyLoss _loss;

    auto _forward(ts::MatrixF const &inputs) -> ts::MatrixF
    {
        return _layer2(_layer1(inputs));
    }
};

auto train(Model &model, ts::PlanarDataset &dataset) -> float
{
    constexpr int epoch_num = 100; // Let's overfit to validate our code
    constexpr float learning_rate = 1e-0;
    float loss = std::numeric_limits<float>::max();

    for (int epoch_i = 0; epoch_i < epoch_num; ++epoch_i) {
        for (auto [inputs, labels] : dataset) {
            loss = model.update(inputs, labels, learning_rate);
            if (epoch_i % 10 == 0)
                std::cout << "epoch: " << epoch_i << "/" << epoch_num << " loss: " << loss
                          << std::endl;
        }
    }
    std::cout << "epoch: " << epoch_num << "/" << epoch_num << " loss: " << loss << std::endl;
    return loss;
}

auto label(Model &model, ts::PlanarDataset &dataset) ->ts::VectorI
{
    std::vector<ts::Tensor<int, 1>> labels;

    for (auto [inputs, _] : dataset) {
        auto batch_labels = model.predict(inputs);
        labels.push_back(batch_labels);
    }
    return ts::concatenate<int, 0>(labels);
}

int main()
{

    ts::PlanarDataset dataset_train("resources/train_planar_data.tsv", true, 300);
    ts::PlanarDataset dataset_test("resources/test_planar_data.tsv", true, 1);
    Model model;

    std::cout << "Training... " << std::endl;
    float loss = train(model, dataset_train);
    std::cout << std::endl;

    std::cout << "Labeling test dataset... ";
    auto labels = label(model, dataset_test);
    std::cout << "done!" << std::endl;

    std::cout << "Dumping labels to 'resources/labels_planar_data.tsv'... ";
    if (std::ofstream output = std::ofstream("resources/labels_planar_data.tsv")) {
        for (auto label : labels) {
            output << label << std::endl;
        }
    }
    std::cout << "done!" << std::endl;
    return 0;
}