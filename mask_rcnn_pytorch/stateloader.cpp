#include "stateloader.h"
#include "debug.h"
#include "nnutils.h"

#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>

#include <iostream>
#include <regex>
#include <stack>

namespace {

enum class ReadState {
  None,
  DictObject,
  ParamName,
  SizeTensorPair,
  TensorSize,
  SizeTensorPairDelim,
  TensorValue,
  List
};

struct DictHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DictHandler> {
  DictHandler() {}

  bool Double(double d) {
    if (current_state_.top() == ReadState::List ||
        current_state_.top() == ReadState::TensorValue) {
      blob_.push_back(static_cast<float>(d));
      ++index_;
    } else {
      throw std::logic_error("Double parsing error");
    }
    return true;
  }

  bool Uint(unsigned u) {
    if (current_state_.top() == ReadState::List ||
        current_state_.top() == ReadState::TensorValue) {
      blob_.push_back(static_cast<float>(u));
      ++index_;
    } else if (current_state_.top() == ReadState::TensorSize) {
      size_.push_back(static_cast<int64_t>(u));
    } else {
      throw std::logic_error("UInt parsing error");
    }
    return true;
  }

  bool Key(const char* str, rapidjson::SizeType length, bool /*copy*/) {
    key_.assign(str, length);
    if (current_state_.top() == ReadState::DictObject) {
      current_state_.push(ReadState::ParamName);
    } else {
      throw std::logic_error("Key parsing error");
    }
    return true;
  }

  bool StartObject() {
    if (current_state_.top() == ReadState::None) {
      current_state_.pop();
      current_state_.push(ReadState::DictObject);
    } else {
      throw std::logic_error("Start object parsing error");
    }
    return true;
  }

  bool EndObject(rapidjson::SizeType /*memberCount*/) {
    if (current_state_.top() != ReadState::DictObject) {
      throw std::logic_error("End object parsing error");
    }
    return true;
  }

  void StartData() {
    current_state_.push(ReadState::TensorValue);
    auto total_length = std::accumulate(size_.begin(), size_.end(), 1,
                                        std::multiplies<int64_t>());
    blob_.resize(static_cast<size_t>(total_length));
    blob_.clear();
    index_ = 0;
  }

  bool StartArray() {
    if (current_state_.top() == ReadState::List) {
      current_state_.push(ReadState::List);
    } else if (current_state_.top() == ReadState::ParamName) {
      current_state_.push(ReadState::SizeTensorPair);
    } else if (current_state_.top() == ReadState::SizeTensorPair) {
      current_state_.push(ReadState::TensorSize);
      size_.clear();
    } else if (current_state_.top() == ReadState::SizeTensorPairDelim) {
      current_state_.pop();
      StartData();
    } else if (current_state_.top() == ReadState::TensorValue) {
      current_state_.push(ReadState::List);
    } else {
      throw std::logic_error("Start array parsing error");
    }
    return true;
  }

  bool EndArray(rapidjson::SizeType elementCount) {
    if (current_state_.top() == ReadState::List) {
      current_state_.pop();
    } else if (current_state_.top() == ReadState::SizeTensorPair) {
      current_state_.pop();
      assert(current_state_.top() == ReadState::ParamName);
      current_state_.pop();
      dict.insert(key_, tensor_);
      std::cout << key_ << " : " << tensor_.type().toString() << " : "
                << tensor_.dim() << "\n";
    } else if (current_state_.top() == ReadState::TensorSize) {
      current_state_.pop();
      if (elementCount == 0) {
        size_.push_back(1);
        StartData();
      } else {
        current_state_.push(ReadState::SizeTensorPairDelim);
      }
    } else if (current_state_.top() == ReadState::TensorValue) {
      current_state_.pop();
      assert(index_ == static_cast<int64_t>(blob_.size()));
      tensor_ = torch::from_blob(blob_.data(), at::IntList(size_),
                                 at::dtype(at::kFloat))
                    .clone();  // clone to copy temp data
      if (blob_.size() == 1) {
        assert(current_state_.top() == ReadState::SizeTensorPair);
        current_state_.pop();
        assert(current_state_.top() == ReadState::ParamName);
        current_state_.pop();
        dict.insert(key_, tensor_);
        std::cout << key_ << " : " << tensor_.type().toString() << " : "
                  << tensor_.dim() << "\n";
      }
    } else {
      throw std::logic_error("End array parsing error");
    }
    return true;
  }

  std::string key_;
  std::vector<int64_t> size_;
  torch::Tensor tensor_;
  std::vector<float> blob_;
  int64_t index_{0};

  std::stack<ReadState> current_state_{{ReadState::None}};

  torch::OrderedDict<std::string, torch::Tensor> dict;
};
}  // namespace

torch::OrderedDict<std::string, torch::Tensor> LoadStateDictJson(
    const std::string& file_name) {
  auto* file = std::fopen(file_name.c_str(), "r");
  if (file) {
    char readBuffer[65536];
    rapidjson::FileReadStream is(file, readBuffer, sizeof(readBuffer));
    rapidjson::Reader reader;
    DictHandler handler;
    auto res = reader.Parse(is, handler);
    std::fclose(file);

    if (!res) {
      throw std::runtime_error(rapidjson::GetParseError_En(res.Code()));
    }

    return handler.dict;
  }
  return torch::OrderedDict<std::string, torch::Tensor>();
}

void LoadStateDictJson(torch::nn::Module& module,
                       const std::string& file_name) {
  // Load weights trained on MS - COCO
  if (file_name.find(".json") != std::string::npos) {
    torch::NoGradGuard no_grad;
    auto new_params = LoadStateDictJson(file_name);
    auto params = module.named_parameters(true /*recurse*/);
    auto buffers = module.named_buffers(true /*recurse*/);

    for (auto& val : new_params) {
      auto name = val.key();
      // fix naming
      auto pos = name.find("running_var");
      if (pos != std::string::npos) {
        name.replace(pos, 11, "running_variance");
      }

      auto* t = params.find(name);
      if (t != nullptr) {
        std::cout << name << " copy\n";
        t->copy_(val.value());
      } else {
        t = buffers.find(name);
        if (t != nullptr) {
          std::cout << name << " copy\n";
          t->copy_(val.value());
        } else {
          // throw std::logic_error(name + " parameter not found!");
          std::cout << name + " parameter not found!\n";
        }
      }
    }

    auto pos = file_name.find_last_of(".");
    std::string new_file_name = file_name.substr(0, pos + 1);
    new_file_name += "dat";
    SaveStateDict(module, new_file_name);
    std::cout << "Model state converted to file :" << new_file_name << "\n";
  } else {
    throw std::invalid_argument("Can't load not a Json file!");
  }
  std::cout.flush();
}

void SaveStateDict(const torch::nn::Module& module,
                   const std::string& file_name) {
  torch::serialize::OutputArchive archive;
  auto params = module.named_parameters(true /*recurse*/);
  auto buffers = module.named_buffers(true /*recurse*/);
  for (const auto& val : params) {
    if (!is_empty(val.value())) {
      archive.write(val.key(), val.value());
    }
  }
  for (const auto& val : buffers) {
    if (!is_empty(val.value())) {
      archive.write(val.key(), val.value(), /*is_buffer*/ true);
    }
  }
  archive.save_to(file_name);
}

void LoadStateDict(torch::nn::Module& module,
                   const std::string& file_name,
                   const std::string& ignore_name_regex) {
  torch::serialize::InputArchive archive;
  archive.load_from(file_name);
  torch::NoGradGuard no_grad;
  std::regex re(ignore_name_regex);
  std::smatch m;
  auto params = module.named_parameters(true /*recurse*/);
  auto buffers = module.named_buffers(true /*recurse*/);
  for (auto& val : params) {
    if (!std::regex_match(val.key(), m, re)) {
      archive.read(val.key(), val.value());
    }
  }
  for (auto& val : buffers) {
    if (!std::regex_match(val.key(), m, re)) {
      archive.read(val.key(), val.value(), /*is_buffer*/ true);
    }
  }
}
