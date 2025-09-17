#ifndef LIDAR_PROCESSOR_SHM_H
#define LIDAR_PROCESSOR_SHM_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>

// System headers for shared memory
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace lidar_processing
{

// Structure to hold semantic class info
struct SemanticClass {
    std::string name;
    std::array<float, 3> color;  // RGB values normalized to [0, 1]
};

/**
 * @brief Shared memory client for communicating with Python inference server
 */
class SharedMemoryClient {
public:
    // Shared memory names (must match Python server)
    static constexpr const char* INPUT_SHM_NAME = "/maskpls_input";
    static constexpr const char* OUTPUT_SHM_NAME = "/maskpls_output";
    static constexpr const char* CONTROL_SHM_NAME = "/maskpls_control";
    
    // Control flags (must match Python server)
    static constexpr int FLAG_IDLE = 0;
    static constexpr int FLAG_NEW_DATA = 1;
    static constexpr int FLAG_PROCESSING = 2;
    static constexpr int FLAG_COMPLETE = 3;
    static constexpr int FLAG_ERROR = 4;
    static constexpr int FLAG_SHUTDOWN = 5;
    
    // Maximum points (must match Python server)
    static constexpr size_t MAX_POINTS = 150000;
    
    SharedMemoryClient();
    ~SharedMemoryClient();
    
    // Initialize shared memory connection
    bool initialize();
    
    // Check if server is running
    bool isServerRunning();
    
    // Wait for server to be ready
    bool waitForServer(int timeout_seconds = 10);
    
    // Process point cloud
    bool processPointCloud(
        const std::vector<float>& points,
        const std::vector<float>& intensities,
        std::vector<int32_t>& semantic_labels,
        std::vector<int32_t>& instance_labels
    );
    
    // Get timing statistics
    void getTimingStats(float& last_time_ms, int& total_frames);
    
    // Shutdown server
    void shutdownServer();
    
private:
    // Shared memory file descriptors
    int fd_input_;
    int fd_output_;
    int fd_control_;
    
    // Mapped memory pointers
    void* input_shm_;
    void* output_shm_;
    void* control_shm_;
    
    // Memory sizes
    size_t input_size_;
    size_t output_size_;
    size_t control_size_;
    
    // Control array pointer
    int32_t* control_array_;
    
    // Connection state
    bool initialized_;
    std::mutex mutex_;
    
    // Internal methods
    bool openSharedMemory();
    void closeSharedMemory();
};

/**
 * @brief ROS2 node for LiDAR processing using shared memory communication
 */
class LidarProcessorSHM : public rclcpp::Node
{
public:
    explicit LidarProcessorSHM(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~LidarProcessorSHM();

private:
    // ROS2 callbacks
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void timerCallback();
    
    // Initialization
    bool initializeSharedMemory();
    void initializeSemanticMapping();
    bool startPythonServer();
    void stopPythonServer();
    
    // Processing functions
    bool convertPointCloud2ToArrays(
        const sensor_msgs::msg::PointCloud2::SharedPtr msg,
        std::vector<float>& points,
        std::vector<float>& intensities
    );
    
    sensor_msgs::msg::PointCloud2 createLabeledPointCloud(
        const std::vector<float>& points,
        const std::vector<int32_t>& semantic_labels,
        const std::vector<int32_t>& instance_labels,
        const std_msgs::msg::Header &header
    );
    
    visualization_msgs::msg::MarkerArray createInstanceMarkers(
        const std::vector<float>& points,
        const std::vector<int32_t>& instance_labels,
        const std_msgs::msg::Header &header
    );
    
    // Helper functions
    std::array<uint8_t, 3> getColorForClass(int class_id);
    std::array<uint8_t, 3> getInstanceColor(int instance_id);
    void filterPointsByBounds(
        std::vector<float>& points,
        std::vector<float>& intensities,
        std::vector<int32_t>& semantic_labels,
        std::vector<int32_t>& instance_labels
    );
    
    // ROS2 interfaces
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr semantic_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr panoptic_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Shared memory client
    std::unique_ptr<SharedMemoryClient> shm_client_;
    
    // Python server process
    std::thread python_server_thread_;
    pid_t python_server_pid_;
    std::atomic<bool> server_running_;
    
    // Parameters
    std::string model_path_;
    std::string config_path_;
    std::string input_topic_;
    bool use_cuda_;
    bool auto_start_server_;
    bool publish_markers_;
    bool verbose_;
    
    // Model configuration
    int num_classes_;
    std::vector<int> things_ids_;
    float overlap_threshold_;
    
    // Spatial bounds
    std::array<float, 2> x_limits_;
    std::array<float, 2> y_limits_;
    std::array<float, 2> z_limits_;
    
    // Performance settings
    int min_points_per_instance_;
    int max_instances_;
    
    // Semantic mapping
    std::unordered_map<int, SemanticClass> semantic_map_;
    
    // Performance monitoring
    std::chrono::steady_clock::time_point last_inference_time_;
    double avg_inference_time_;
    int inference_count_;
    int error_count_;
};

}  // namespace lidar_processing

#endif  // LIDAR_PROCESSOR_SHM_H
