#pragma once

#include <fstream>

#include "proto/array.pb.h"
#include "layer_base.hpp"

namespace ts {
template <typename T>
class Saver {
  public:
    LayerBase<T> & _model_base;
    proto::Model _proto_model;
    explicit Saver(LayerBase<T> &model) : _model_base(model), _proto_model() {}

    auto save(std::string const & output_path) -> void
    {
        _proto_model.clear_variables();
        for (auto const &param : _model_base.parameters()) {
            auto *var = _proto_model.add_variables();
            for (auto v : *param.get().get()) {
                var->add_values(v);
            }
        }

        std::fstream output(output_path.c_str(),
                            std::ios::out | std::ios::trunc | std::ios::binary);
        _proto_model.SerializePartialToOstream(&output);
    }

    auto load(std::string const & input_path) -> void
    {
        std::fstream input(input_path.c_str(), std::ios::in | std::ios::binary);
        if (!_proto_model.ParseFromIstream(&input)) {
            std::cerr << "Failed to parse serialized array." << std::endl;
        }

        auto &params = _model_base.parameters();

        if (params.size() != _proto_model.variables_size()) {
            std::cerr << "Saved doesn't have the same about of variables." << std::endl;
        }

        for (int i = 0; i < params.size(); ++i) {
            const auto &var = _proto_model.variables(i);
            auto &buffer = *params[i].get().get();
            if (buffer.size() != var.values_size()) {
                std::cerr << "Model variable size is different than the loaded one. ";
                std::cerr << buffer.size() << " != " << var.values_size() << std::endl;
            }
            for (int j = 0; j < buffer.size(); ++j) {
                buffer[j] = var.values(j);
            }
        }
    }
};
} // namespace ts
