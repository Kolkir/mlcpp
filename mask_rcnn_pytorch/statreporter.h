#ifndef STATREPORTER_H
#define STATREPORTER_H

#include <atomic>
#include <condition_variable>
#include <iosfwd>
#include <mutex>
#include <string>
#include <thread>

enum class PrintState {
  StartEpoch,
  ReportEpoch,
  ReportTrainStep,
  ReportValidationStep,
  Idle,
  Stop
};

struct LossStat {
  float loss{0};
  float loss_rpn_class{0};
  float loss_rpn_bbox{0};
  float loss_mrcnn_class{0};
  float loss_mrcnn_bbox{0};
  float loss_mrcnn_mask{0};
};

class StatReporter {
 public:
  StatReporter(uint32_t epochs_num,
               uint32_t train_steps_num,
               uint32_t val_steps_num);
  StatReporter(const StatReporter&) = delete;
  StatReporter& operator=(const StatReporter&) = delete;
  ~StatReporter();

  void StartEpoch(uint32_t i, double learning_rate);
  void ReportTrainStep(uint32_t i, const LossStat& stat);
  void ReportValidationStep(uint32_t i, const LossStat& stat);
  void ReportEpoch(const LossStat& train_stat, const LossStat& valid_stat);

  void Stop();

 private:
  void PrintLoss(std::ostream& out, const LossStat& stat);
  void PrintLossSmall(std::ostream& out, const LossStat& stat);
  void PrintLoop();
  void ClearStepScreen();

 private:
  uint32_t epochs_num_{0};
  uint32_t epoch_{0};
  double learning_rate_{0.0};
  LossStat train_stat_;
  LossStat valid_stat_;

  uint32_t train_steps_num_{0};
  uint32_t train_step_{0};
  uint32_t val_steps_num_{0};
  uint32_t val_step_{0};

  std::thread print_thread;
  PrintState print_state_{PrintState::Idle};
  std::mutex state_mutex_;
  std::condition_variable state_cv_;
};

#endif  // STATREPORTER_H
