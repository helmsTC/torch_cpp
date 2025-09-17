#include "lidar/lidar_shm.h"
#include <memory>
#include <chrono>
#include <random>
#include <signal.h>
#include <sys/wait.h>

namespace lidar_processing
{

// ============================================================================
// SharedMemoryClient Implementation
// ============================================================================

SharedMemoryClient::SharedMemoryClient()
    : fd_input_(-1), fd_output_(-1), fd_control_(-1),
      input_shm_(nullptr), output_shm_(nullptr), control_shm_(nullptr),
      control_array_(nullptr), initialized_(false)
{
    // Calculate memory sizes
    input_size_ = MAX_POINTS * 4 * sizeof(float);  // N x 4 floats
    output_size_ = MAX_POINTS * 2 * sizeof(int32_t);  // N x 2 ints
    control_size_ = 32;  // Control structure (8 int32s)
}

SharedMemoryClient::~SharedMemoryClient()
{
    closeSharedMemory();
}

bool SharedMemoryClient::initialize()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        return true;
    }
    
    if (!openSharedMemory()) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool SharedMemoryClient::openSharedMemory()
{
    // Open shared memory segments created by Python server
    fd_input_ = shm_open(INPUT_SHM_NAME, O_RDWR, 0666);
    if (fd_input_ == -1) {
        std::cerr << "Failed to open input shared memory: " << strerror(errno) << std::endl;
        return false;
    }
    
    fd_output_ = shm_open(OUTPUT_SHM_NAME, O_RDONLY, 0666);
    if (fd_output_ == -1) {
        std::cerr << "Failed to open output shared memory: " << strerror(errno) << std::endl;
        close(fd_input_);
        return false;
    }
    
    fd_control_ = shm_open(CONTROL_SHM_NAME, O_RDWR, 0666);
    if (fd_control_ == -1) {
        std::cerr << "Failed to open control shared memory: " << strerror(errno) << std::endl;
        close(fd_input_);
        close(fd_output_);
        return false;
    }
    
    // Map memory segments
    input_shm_ = mmap(nullptr, input_size_, PROT_READ | PROT_WRITE, 
                     MAP_SHARED, fd_input_, 0);
    if (input_shm_ == MAP_FAILED) {
        std::cerr << "Failed to map input memory: " << strerror(errno) << std::endl;
        closeSharedMemory();
        return false;
    }
    
    output_shm_ = mmap(nullptr, output_size_, PROT_READ, 
                      MAP_SHARED, fd_output_, 0);
    if (output_shm_ == MAP_FAILED) {
        std::cerr << "Failed to map output memory: " << strerror(errno) << std::endl;
        closeSharedMemory();
        return false;
    }
    
    control_shm_ = mmap(nullptr, control_size_, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd_control_, 0);
    if (control_shm_ == MAP_FAILED) {
        std::cerr << "Failed to map control memory: " << strerror(errno) << std::endl;
        closeSharedMemory();
        return false;
    }
    
    // Get control array pointer
    control_array_ = static_cast<int32_t*>(control_shm_);
    
    return true;
}

void SharedMemoryClient::closeSharedMemory()
{
    if (input_shm_ != nullptr && input_shm_ != MAP_FAILED) {
        munmap(input_shm_, input_size_);
        input_shm_ = nullptr;
    }
    
    if (output_shm_ != nullptr && output_shm_ != MAP_FAILED) {
        munmap(output_shm_, output_size_);
        output_shm_ = nullptr;
    }
    
    if (control_shm_ != nullptr && control_shm_ != MAP_FAILED) {
        munmap(control_shm_, control_size_);
        control_shm_ = nullptr;
    }
    
    if (fd_input_ != -1) {
        close(fd_input_);
        fd_input_ = -1;
    }
    
    if (fd_output_ != -1) {
        close(fd_output_);
        fd_output_ = -1;
    }
    
    if (fd_control_ != -1) {
        close(fd_control_);
        fd_control_ = -1;
    }
    
    control_array_ = nullptr;
    initialized_ = false;
}

bool SharedMemoryClient::isServerRunning()
{
    if (!initialized_) {
        return false;
    }
    
    // Check if server is responsive
    int flag = control_array_[0];
    return (flag != FLAG_SHUTDOWN && flag != FLAG_ERROR);
}

bool SharedMemoryClient::waitForServer(int timeout_seconds)
{
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        if (initialize() && isServerRunning()) {
            return true;
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= timeout_seconds) {
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool SharedMemoryClient::processPointCloud(
    const std::vector<float>& points,
    const std::vector<float>& intensities,
    std::vector<int32_t>& semantic_labels,
    std::vector<int32_t>& instance_labels)
{
    if (!initialized_ || !isServerRunning()) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t num_points = points.size() / 3;
    
    if (num_points == 0 || num_points > MAX_POINTS) {
        std::cerr << "Invalid number of points: " << num_points << std::endl;
        return false;
    }
    
    // Wait for server to be idle
    int max_wait = 100;  // 10 seconds (100 * 100ms)
    while (control_array_[0] != FLAG_IDLE && 
           control_array_[0] != FLAG_COMPLETE && 
           max_wait > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        max_wait--;
    }
    
    if (max_wait == 0) {
        std::cerr << "Timeout waiting for server to be ready" << std::endl;
        return false;
    }
    
    // Write input data to shared memory
    float* input_ptr = static_cast<float*>(input_shm_);
    
    for (size_t i = 0; i < num_points; ++i) {
        // Write x, y, z, intensity
        input_ptr[i * 4 + 0] = points[i * 3 + 0];
        input_ptr[i * 4 + 1] = points[i * 3 + 1];
        input_ptr[i * 4 + 2] = points[i * 3 + 2];
        input_ptr[i * 4 + 3] = (i < intensities.size()) ? intensities[i] : 0.0f;
    }
    
    // Set control flags
    control_array_[1] = static_cast<int32_t>(num_points);  // Set number of points
    control_array_[0] = FLAG_NEW_DATA;  // Signal new data available
    
    // Wait for processing to complete
    auto start_time = std::chrono::steady_clock::now();
    int timeout_ms = 5000;  // 5 second timeout
    
    while (true) {
        int flag = control_array_[0];
        
        if (flag == FLAG_COMPLETE) {
            // Read output data
            int32_t* output_ptr = static_cast<int32_t*>(output_shm_);
            
            semantic_labels.resize(num_points);
            instance_labels.resize(num_points);
            
            for (size_t i = 0; i < num_points; ++i) {
                semantic_labels[i] = output_ptr[i * 2 + 0];
                instance_labels[i] = output_ptr[i * 2 + 1];
            }
            
            // Reset flag
            control_array_[0] = FLAG_IDLE;
            
            return true;
        }
        else if (flag == FLAG_ERROR) {
            std::cerr << "Server reported error during processing" << std::endl;
            control_array_[0] = FLAG_IDLE;
            return false;
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {
            std::cerr << "Timeout waiting for processing to complete" << std::endl;
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void SharedMemoryClient::getTimingStats(float& last_time_ms, int& total_frames)
{
    if (!initialized_) {
        last_time_ms = 0.0f;
        total_frames = 0;
        return;
    }
    
    last_time_ms = static_cast<float>(control_array_[2]);
    total_frames = control_array_[3];
}

void SharedMemoryClient::shutdownServer()
{
    if (initialized_ && control_array_ != nullptr) {
        control_array_[0] = FLAG_SHUTDOWN;
    }
}

// ============================================================================
// LidarProcessorSHM Implementation
// ============================================================================

LidarProcessorSHM::LidarProcessorSHM(const rclcpp::NodeOptions &options)
    : Node("lidar_processor_shm", options),
      python_server_pid_(-1),
      server_running_(false),
      num_classes_(20),
      overlap_threshold_(0.8),
      avg_inference_time_(0.0),
      inference_count_(0),
      error_count_(0)
{
    // Declare and get parameters
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("config_path", "");
    this->declare_parameter<std::string>("input_topic", "/velodyne_points");
    this->declare_parameter<bool>("use_cuda", true);
    this->declare_parameter<bool>("auto_start_server", true);
    this->declare_parameter<bool>("publish_markers", true);
    this->declare_parameter<bool>("verbose", false);
    this->declare_parameter<int>("num_classes", 20);
    this->declare_parameter<double>("overlap_threshold", 0.8);
    
    // Spatial bounds
    this->declare_parameter<std::vector<double>>("x_limits", {-48.0, 48.0});
    this->declare_parameter<std::vector<double>>("y_limits", {-48.0, 48.0});
    this->declare_parameter<std::vector<double>>("z_limits", {-4.0, 1.5});
    
    // Things IDs
    this->declare_parameter<std::vector<int64_t>>("things_ids", 
        std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8});
    
    // Performance settings
    this->declare_parameter<int>("min_points_per_instance", 50);
    this->declare_parameter<int>("max_instances", 100);
    
    // Get parameters
    model_path_ = this->get_parameter("model_path").as_string();
    config_path_ = this->get_parameter("config_path").as_string();
    input_topic_ = this->get_parameter("input_topic").as_string();
    use_cuda_ = this->get_parameter("use_cuda").as_bool();
    auto_start_server_ = this->get_parameter("auto_start_server").as_bool();
    publish_markers_ = this->get_parameter("publish_markers").as_bool();
    verbose_ = this->get_parameter("verbose").as_bool();
    num_classes_ = this->get_parameter("num_classes").as_int();
    overlap_threshold_ = this->get_parameter("overlap_threshold").as_double();
    
    // Get spatial bounds
    auto x_limits = this->get_parameter("x_limits").as_double_array();
    auto y_limits = this->get_parameter("y_limits").as_double_array();
    auto z_limits = this->get_parameter("z_limits").as_double_array();
    
    x_limits_[0] = x_limits[0];
    x_limits_[1] = x_limits[1];
    y_limits_[0] = y_limits[0];
    y_limits_[1] = y_limits[1];
    z_limits_[0] = z_limits[0];
    z_limits_[1] = z_limits[1];
    
    // Get things IDs
    auto things_ids_param = this->get_parameter("things_ids").as_integer_array();
    things_ids_.clear();
    for (auto id : things_ids_param) {
        things_ids_.push_back(static_cast<int>(id));
    }
    
    // Get performance settings
    min_points_per_instance_ = this->get_parameter("min_points_per_instance").as_int();
    max_instances_ = this->get_parameter("max_instances").as_int();
    
    // Initialize semantic mapping
    initializeSemanticMapping();
    
    // Create shared memory client
    shm_client_ = std::make_unique<SharedMemoryClient>();
    
    // Start Python server if requested
    if (auto_start_server_) {
        if (!model_path_.empty()) {
            if (startPythonServer()) {
                RCLCPP_INFO(this->get_logger(), "Python inference server started");
            } else {
                RCLCPP_ERROR(this->get_logger(), "Failed to start Python inference server");
            }
        } else {
            RCLCPP_WARN(this->get_logger(), 
                       "No model path provided. Cannot auto-start server.");
        }
    }
    
    // Initialize shared memory connection
    if (!initializeSharedMemory()) {
        RCLCPP_ERROR(this->get_logger(), 
                    "Failed to initialize shared memory. Make sure Python server is running.");
    }
    
    // Create subscriber
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&LidarProcessorSHM::pointcloudCallback, this, std::placeholders::_1));
    
    // Create publishers
    semantic_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/semantic_points", rclcpp::SensorDataQoS());
    
    panoptic_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/panoptic_points", rclcpp::SensorDataQoS());
    
    if (publish_markers_) {
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/instance_markers", 10);
    }
    
    // Create timer for status updates
    timer_ = this->create_wall_timer(
        std::chrono::seconds(5),
        std::bind(&LidarProcessorSHM::timerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), 
               "LidarProcessorSHM initialized. Subscribing to: %s", 
               input_topic_.c_str());
}

LidarProcessorSHM::~LidarProcessorSHM()
{
    stopPythonServer();
}

void LidarProcessorSHM::initializeSemanticMapping()
{
    // Semantic-KITTI color mapping (BGR -> RGB, normalized)
    semantic_map_[0] = {"unlabeled", {0.0f, 0.0f, 0.0f}};
    semantic_map_[1] = {"car", {0.39f, 0.59f, 0.96f}};
    semantic_map_[2] = {"bicycle", {0.96f, 0.90f, 0.39f}};
    semantic_map_[3] = {"motorcycle", {0.12f, 0.24f, 0.59f}};
    semantic_map_[4] = {"truck", {0.31f, 0.12f, 0.71f}};
    semantic_map_[5] = {"other-vehicle", {0.0f, 0.0f, 1.0f}};
    semantic_map_[6] = {"person", {1.0f, 0.12f, 0.12f}};
    semantic_map_[7] = {"bicyclist", {1.0f, 0.16f, 0.78f}};
    semantic_map_[8] = {"motorcyclist", {0.59f, 0.12f, 0.35f}};
    semantic_map_[9] = {"road", {1.0f, 0.0f, 1.0f}};
    semantic_map_[10] = {"parking", {1.0f, 0.59f, 1.0f}};
    semantic_map_[11] = {"sidewalk", {0.29f, 0.0f, 0.29f}};
    semantic_map_[12] = {"other-ground", {0.29f, 0.0f, 0.69f}};
    semantic_map_[13] = {"building", {1.0f, 0.78f, 0.0f}};
    semantic_map_[14] = {"fence", {0.98f, 0.47f, 0.20f}};
    semantic_map_[15] = {"vegetation", {0.0f, 0.69f, 0.0f}};
    semantic_map_[16] = {"trunk", {0.53f, 0.24f, 0.0f}};
    semantic_map_[17] = {"terrain", {0.59f, 0.94f, 0.31f}};
    semantic_map_[18] = {"pole", {1.0f, 0.94f, 0.59f}};
    semantic_map_[19] = {"traffic-sign", {1.0f, 0.0f, 0.0f}};
    
    RCLCPP_INFO(this->get_logger(), 
               "Initialized semantic mapping with %zu classes", 
               semantic_map_.size());
}

bool LidarProcessorSHM::initializeSharedMemory()
{
    if (!shm_client_->waitForServer(10)) {
        RCLCPP_WARN(this->get_logger(), 
                   "Python server not ready. Waiting for server to start...");
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "Connected to Python inference server");
    return true;
}

bool LidarProcessorSHM::startPythonServer()
{
    // Get the path to the Python script
    std::string script_path = "maskpls_inference_server.py";
    
    // Build command
    std::stringstream cmd;
    cmd << "python3 " << script_path;
    cmd << " --model " << model_path_;
    
    if (!config_path_.empty()) {
        cmd << " --config " << config_path_;
    }
    
    if (use_cuda_) {
        cmd << " --cuda";
    }
    
    if (verbose_) {
        cmd << " --verbose";
    }
    
    RCLCPP_INFO(this->get_logger(), "Starting Python server: %s", cmd.str().c_str());
    
    // Fork and execute Python server
    python_server_pid_ = fork();
    
    if (python_server_pid_ == 0) {
        // Child process - execute Python server
        execl("/bin/sh", "sh", "-c", cmd.str().c_str(), nullptr);
        exit(1);  // If exec fails
    }
    else if (python_server_pid_ < 0) {
        RCLCPP_ERROR(this->get_logger(), "Failed to fork process");
        return false;
    }
    
    // Parent process - wait for server to be ready
    server_running_ = true;
    
    // Give server time to initialize
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Wait for server to be ready
    if (!shm_client_->waitForServer(30)) {
        RCLCPP_ERROR(this->get_logger(), "Python server failed to start");
        stopPythonServer();
        return false;
    }
    
    return true;
}

void LidarProcessorSHM::stopPythonServer()
{
    if (server_running_) {
        // Send shutdown signal via shared memory
        shm_client_->shutdownServer();
        
        // Give server time to shut down gracefully
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Kill process if still running
        if (python_server_pid_ > 0) {
            kill(python_server_pid_, SIGTERM);
            
            // Wait for process to exit
            int status;
            waitpid(python_server_pid_, &status, 0);
            
            python_server_pid_ = -1;
        }
        
        server_running_ = false;
        RCLCPP_INFO(this->get_logger(), "Python server stopped");
    }
}

void LidarProcessorSHM::pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    if (!shm_client_->isServerRunning()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                            "Python inference server not running. Skipping processing.");
        error_count_++;
        
        // Try to reconnect
        if (error_count_ > 10) {
            initializeSharedMemory();
            error_count_ = 0;
        }
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    RCLCPP_DEBUG(this->get_logger(), 
                "Received pointcloud with %d points", 
                msg->width * msg->height);
    
    // Convert PointCloud2 to arrays
    std::vector<float> points;
    std::vector<float> intensities;
    
    if (!convertPointCloud2ToArrays(msg, points, intensities)) {
        RCLCPP_WARN(this->get_logger(), "Failed to convert point cloud");
        return;
    }
    
    size_t num_points = points.size() / 3;
    
    if (num_points == 0) {
        RCLCPP_WARN(this->get_logger(), "Empty point cloud");
        return;
    }
    
    // Process with Python server via shared memory
    std::vector<int32_t> semantic_labels;
    std::vector<int32_t> instance_labels;
    
    if (!shm_client_->processPointCloud(points, intensities, 
                                        semantic_labels, instance_labels)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to process point cloud");
        error_count_++;
        return;
    }
    
    // Filter by spatial bounds (optional)
    filterPointsByBounds(points, intensities, semantic_labels, instance_labels);
    
    // Create and publish labeled point cloud
    auto labeled_cloud = createLabeledPointCloud(
        points, semantic_labels, instance_labels, msg->header);
    
    semantic_pub_->publish(labeled_cloud);
    
    // Publish instance markers if enabled
    if (publish_markers_ && marker_pub_) {
        auto markers = createInstanceMarkers(
            points, instance_labels, msg->header);
        marker_pub_->publish(markers);
    }
    
    // Update performance statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    inference_count_++;
    avg_inference_time_ = (avg_inference_time_ * (inference_count_ - 1) + 
                          duration.count()) / inference_count_;
    
    if (verbose_) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Processing time: %ld ms (avg: %.2f ms), Points: %zu", 
                            duration.count(), avg_inference_time_, num_points);
    }
    
    error_count_ = 0;  // Reset error count on success
}

void LidarProcessorSHM::timerCallback()
{
    // Get timing statistics from server
    float last_time_ms;
    int total_frames;
    shm_client_->getTimingStats(last_time_ms, total_frames);
    
    if (verbose_ && total_frames > 0) {
        RCLCPP_INFO(this->get_logger(), 
                   "Status - Total frames: %d, Last inference: %.1f ms, Avg: %.2f ms", 
                   total_frames, last_time_ms, avg_inference_time_);
    }
    
    // Check server health
    if (!shm_client_->isServerRunning()) {
        RCLCPP_WARN(this->get_logger(), "Python server not responding");
        
        // Try to restart if auto-start is enabled
        if (auto_start_server_ && !model_path_.empty()) {
            RCLCPP_INFO(this->get_logger(), "Attempting to restart Python server...");
            stopPythonServer();
            startPythonServer();
        }
    }
}

bool LidarProcessorSHM::convertPointCloud2ToArrays(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg,
    std::vector<float>& points,
    std::vector<float>& intensities)
{
    // Get field offsets
    int x_offset = -1, y_offset = -1, z_offset = -1, intensity_offset = -1;
    
    for (const auto& field : msg->fields) {
        if (field.name == "x") x_offset = field.offset;
        else if (field.name == "y") y_offset = field.offset;
        else if (field.name == "z") z_offset = field.offset;
        else if (field.name == "intensity" || field.name == "i") 
            intensity_offset = field.offset;
    }
    
    if (x_offset == -1 || y_offset == -1 || z_offset == -1) {
        RCLCPP_ERROR(this->get_logger(), "PointCloud2 missing required xyz fields");
        return false;
    }
    
    size_t num_points = msg->width * msg->height;
    int point_step = msg->point_step;
    
    // Reserve space
    points.reserve(num_points * 3);
    intensities.reserve(num_points);
    
    // Extract point data
    for (size_t i = 0; i < num_points; ++i) {
        const uint8_t* point_data = &msg->data[i * point_step];
        
        float x = *reinterpret_cast<const float*>(point_data + x_offset);
        float y = *reinterpret_cast<const float*>(point_data + y_offset);
        float z = *reinterpret_cast<const float*>(point_data + z_offset);
        
        points.push_back(x);
        points.push_back(y);
        points.push_back(z);
        
        if (intensity_offset != -1) {
            float intensity = *reinterpret_cast<const float*>(
                point_data + intensity_offset);
            intensities.push_back(intensity);
        } else {
            intensities.push_back(0.0f);
        }
    }
    
    return true;
}

sensor_msgs::msg::PointCloud2 LidarProcessorSHM::createLabeledPointCloud(
    const std::vector<float>& points,
    const std::vector<int32_t>& semantic_labels,
    const std::vector<int32_t>& instance_labels,
    const std_msgs::msg::Header &header)
{
    sensor_msgs::msg::PointCloud2 msg;
    msg.header = header;
    
    size_t num_points = points.size() / 3;
    
    // Setup message structure with XYZRGBL (L for label)
    msg.height = 1;
    msg.width = num_points;
    msg.is_bigendian = false;
    msg.is_dense = true;
    
    // Define fields
    sensor_msgs::msg::PointField field_x, field_y, field_z, field_rgb, field_label;
    
    field_x.name = "x";
    field_x.offset = 0;
    field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_x.count = 1;
    
    field_y.name = "y";
    field_y.offset = 4;
    field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_y.count = 1;
    
    field_z.name = "z";
    field_z.offset = 8;
    field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_z.count = 1;
    
    field_rgb.name = "rgb";
    field_rgb.offset = 12;
    field_rgb.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field_rgb.count = 1;
    
    field_label.name = "label";
    field_label.offset = 16;
    field_label.datatype = sensor_msgs::msg::PointField::UINT32;
    field_label.count = 1;
    
    msg.fields = {field_x, field_y, field_z, field_rgb, field_label};
    msg.point_step = 20;  // 5 * 4 bytes
    msg.row_step = msg.point_step * msg.width;
    msg.data.resize(msg.row_step);
    
    // Fill point cloud data
    for (size_t i = 0; i < num_points; ++i) {
        uint8_t* point_data = &msg.data[i * msg.point_step];
        
        // Position
        *reinterpret_cast<float*>(point_data + 0) = points[i * 3 + 0];
        *reinterpret_cast<float*>(point_data + 4) = points[i * 3 + 1];
        *reinterpret_cast<float*>(point_data + 8) = points[i * 3 + 2];
        
        // Color based on semantic class
        int class_id = (i < semantic_labels.size()) ? semantic_labels[i] : 0;
        auto color = getColorForClass(class_id);
        
        // Pack RGB into float (PCL format)
        uint32_t rgb = ((uint32_t)color[0] << 16 | 
                       (uint32_t)color[1] << 8 | 
                       (uint32_t)color[2]);
        *reinterpret_cast<float*>(point_data + 12) = *reinterpret_cast<float*>(&rgb);
        
        // Label (combine semantic and instance)
        uint32_t label = ((uint32_t)semantic_labels[i] << 16) | 
                        ((uint32_t)instance_labels[i] & 0xFFFF);
        *reinterpret_cast<uint32_t*>(point_data + 16) = label;
    }
    
    return msg;
}

visualization_msgs::msg::MarkerArray LidarProcessorSHM::createInstanceMarkers(
    const std::vector<float>& points,
    const std::vector<int32_t>& instance_labels,
    const std_msgs::msg::Header &header)
{
    visualization_msgs::msg::MarkerArray markers;
    
    // Group points by instance
    std::unordered_map<int32_t, std::vector<size_t>> instance_points;
    
    for (size_t i = 0; i < instance_labels.size(); ++i) {
        if (instance_labels[i] > 0) {  // Skip background (0)
            instance_points[instance_labels[i]].push_back(i);
        }
    }
    
    int marker_id = 0;
    
    for (const auto& [instance_id, point_indices] : instance_points) {
        if (point_indices.size() < min_points_per_instance_) {
            continue;  // Skip small instances
        }
        
        if (marker_id >= max_instances_) {
            break;  // Limit number of markers
        }
        
        // Compute bounding box
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        float max_z = std::numeric_limits<float>::lowest();
        
        for (size_t idx : point_indices) {
            float x = points[idx * 3 + 0];
            float y = points[idx * 3 + 1];
            float z = points[idx * 3 + 2];
            
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
            min_z = std::min(min_z, z);
            max_x = std::max(max_x, x);
            max_y = std::max(max_y, y);
            max_z = std::max(max_z, z);
        }
        
        // Create bounding box marker
        visualization_msgs::msg::Marker marker;
        marker.header = header;
        marker.ns = "instances";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = (min_x + max_x) / 2.0;
        marker.pose.position.y = (min_y + max_y) / 2.0;
        marker.pose.position.z = (min_z + max_z) / 2.0;
        
        marker.scale.x = max_x - min_x;
        marker.scale.y = max_y - min_y;
        marker.scale.z = max_z - min_z;
        
        auto color = getInstanceColor(instance_id);
        marker.color.r = color[0] / 255.0f;
        marker.color.g = color[1] / 255.0f;
        marker.color.b = color[2] / 255.0f;
        marker.color.a = 0.5f;  // Semi-transparent
        
        marker.lifetime = rclcpp::Duration(0, 100000000);  // 0.1 seconds
        
        markers.markers.push_back(marker);
    }
    
    return markers;
}

std::array<uint8_t, 3> LidarProcessorSHM::getColorForClass(int class_id)
{
    if (semantic_map_.find(class_id) != semantic_map_.end()) {
        auto &color = semantic_map_[class_id].color;
        return {
            static_cast<uint8_t>(color[0] * 255),
            static_cast<uint8_t>(color[1] * 255),
            static_cast<uint8_t>(color[2] * 255)
        };
    }
    // Default color (gray)
    return {128, 128, 128};
}

std::array<uint8_t, 3> LidarProcessorSHM::getInstanceColor(int instance_id)
{
    // Generate distinct colors for instances using golden ratio
    float golden_ratio = 0.618033988749895;
    float hue = fmod(instance_id * golden_ratio, 1.0);
    
    // HSV to RGB conversion
    float h = hue * 6.0;
    float c = 1.0;
    float x = c * (1 - fabs(fmod(h, 2) - 1));
    
    float r, g, b;
    if (h < 1) { r = c; g = x; b = 0; }
    else if (h < 2) { r = x; g = c; b = 0; }
    else if (h < 3) { r = 0; g = c; b = x; }
    else if (h < 4) { r = 0; g = x; b = c; }
    else if (h < 5) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    
    return {
        static_cast<uint8_t>(r * 255),
        static_cast<uint8_t>(g * 255),
        static_cast<uint8_t>(b * 255)
    };
}

void LidarProcessorSHM::filterPointsByBounds(
    std::vector<float>& points,
    std::vector<float>& intensities,
    std::vector<int32_t>& semantic_labels,
    std::vector<int32_t>& instance_labels)
{
    std::vector<float> filtered_points;
    std::vector<float> filtered_intensities;
    std::vector<int32_t> filtered_semantic;
    std::vector<int32_t> filtered_instance;
    
    size_t num_points = points.size() / 3;
    
    for (size_t i = 0; i < num_points; ++i) {
        float x = points[i * 3 + 0];
        float y = points[i * 3 + 1];
        float z = points[i * 3 + 2];
        
        // Check bounds
        if (x >= x_limits_[0] && x <= x_limits_[1] &&
            y >= y_limits_[0] && y <= y_limits_[1] &&
            z >= z_limits_[0] && z <= z_limits_[1]) {
            
            filtered_points.push_back(x);
            filtered_points.push_back(y);
            filtered_points.push_back(z);
            
            if (i < intensities.size()) {
                filtered_intensities.push_back(intensities[i]);
            }
            
            if (i < semantic_labels.size()) {
                filtered_semantic.push_back(semantic_labels[i]);
            }
            
            if (i < instance_labels.size()) {
                filtered_instance.push_back(instance_labels[i]);
            }
        }
    }
    
    // Replace with filtered data
    points = std::move(filtered_points);
    intensities = std::move(filtered_intensities);
    semantic_labels = std::move(filtered_semantic);
    instance_labels = std::move(filtered_instance);
}

}  // namespace lidar_processing

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(lidar_processing::LidarProcessorSHM)
