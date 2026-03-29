// csv_logger.hpp
// Drop this file alongside nano_gpt.cpp and #include it.
// Logs per-step training metrics to a CSV for later plotting.

#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

class CSVLogger {
public:
    // run_label: e.g. "adamw_1dev", "adamw_4dev_tp" — appears as a column so
    // multiple CSVs can be concatenated and plotted together later.
    explicit CSVLogger(const std::string &path, const std::string &run_label = "adamw") :
        path_(path), run_label_(run_label), wall_start_(std::chrono::high_resolution_clock::now()) {
        bool exists = std::filesystem::exists(path_);
        file_.open(path_, std::ios::app);
        if (!file_.is_open()) {
            throw std::runtime_error("CSVLogger: cannot open file: " + path_);
        }
        // Write header only for new files
        if (!exists || std::filesystem::file_size(path_) == 0) {
            file_ << "run_label,step,wall_time_s,step_time_ms,optimizer_time_ms,train_loss\n";
            file_.flush();
        }
    }

    // Call once per optimizer step (i.e. after scheduler->step())
    void log(uint32_t step, double step_time_ms, double optimizer_time_ms, float train_loss) {
        double wall_s = elapsed_seconds();
        file_ << run_label_ << "," << step << "," << wall_s << "," << step_time_ms << "," << optimizer_time_ms << ","
              << train_loss << "\n";
        file_.flush();  // flush every row so you can tail -f during a run
    }

    double elapsed_seconds() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - wall_start_).count();
    }

    ~CSVLogger() {
        if (file_.is_open()) {
            file_.close();
        }
    }

private:
    std::string path_;
    std::string run_label_;
    std::chrono::high_resolution_clock::time_point wall_start_;
    std::ofstream file_;
};
