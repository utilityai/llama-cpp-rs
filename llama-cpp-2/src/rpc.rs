//! RPC backend support for distributed inference
//!
//! This module provides support for running inference across multiple machines
//! using the RPC (Remote Procedure Call) backend from llama.cpp's GGML library.
//!
//! # Features
//!
//! - Distributed model execution across multiple nodes
//! - Remote GPU/CPU resource utilization
//! - Network-based tensor operations
//!
//! # Example
//!
//! ```no_run
//! use llama_cpp_2::rpc::RpcDevice;
//! use llama_cpp_2::model::params::LlamaModelParams;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Add an RPC device
//! let device = RpcDevice::add("192.168.1.100:50052")?;
//!
//! // Use the device in model parameters
//! let params = LlamaModelParams::default()
//!     .with_split_mode(llama_cpp_2::model::params::LlamaSplitMode::Row);
//! # Ok(())
//! # }
//! ```

use std::ffi::{CStr, CString};
use std::ptr::NonNull;

/// Errors that can occur when working with RPC backend
#[derive(thiserror::Error, Debug)]
pub enum RpcError {
    /// Failed to convert string to C string
    #[error("Failed to convert string: {0}")]
    StringConversion(#[from] std::ffi::NulError),
    
    /// Failed to initialize RPC backend
    #[error("Failed to initialize RPC backend for endpoint: {endpoint}")]
    InitializationFailed { endpoint: String },
    
    /// Failed to add RPC device
    #[error("Failed to add RPC device: {endpoint}")]
    AddDeviceFailed { endpoint: String },
    
    /// Failed to query device memory
    #[error("Failed to query device memory")]
    MemoryQueryFailed,
    
    /// Failed to start RPC server
    #[error("Failed to start RPC server on endpoint: {endpoint}")]
    ServerStartFailed { endpoint: String },
}

/// RPC backend for distributed inference across multiple machines
#[derive(Debug)]
pub struct RpcBackend {
    backend: NonNull<llama_cpp_sys_2::ggml_backend>,
    endpoint: String,
}

impl RpcBackend {
    /// Initialize a new RPC backend for the given endpoint
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The RPC server endpoint (e.g., "127.0.0.1:50052")
    ///
    /// # Returns
    ///
    /// Returns `Ok(RpcBackend)` on success.
    ///
    /// # Errors
    ///
    /// * `StringConversion` - Endpoint contains null bytes
    /// * `InitializationFailed` - Backend initialization failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_2::rpc::RpcBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let backend = RpcBackend::init("127.0.0.1:50052")?;
    /// println!("Connected to RPC endpoint: {}", backend.endpoint());
    /// # Ok(())
    /// # }
    /// ```
    pub fn init(endpoint: &str) -> Result<Self, RpcError> {
        let c_endpoint = CString::new(endpoint)?;
        
        let backend = unsafe {
            llama_cpp_sys_2::ggml_backend_rpc_init(c_endpoint.as_ptr())
        };
        
        NonNull::new(backend)
            .map(|ptr| Self {
                backend: ptr,
                endpoint: endpoint.to_string(),
            })
            .ok_or_else(|| RpcError::InitializationFailed {
                endpoint: endpoint.to_string(),
            })
    }
    
    /// Check if this backend is an RPC backend
    ///
    /// # Returns
    ///
    /// Returns `true` if this is an RPC backend, `false` otherwise.
    #[must_use]
    pub fn is_rpc(&self) -> bool {
        unsafe { llama_cpp_sys_2::ggml_backend_is_rpc(self.backend.as_ptr()) }
    }
    
    /// Get the buffer type for this RPC backend
    ///
    /// # Returns
    ///
    /// Returns the buffer type if available.
    #[must_use]
    pub fn buffer_type(&self) -> Option<NonNull<llama_cpp_sys_2::ggml_backend_buffer_type>> {
        let c_endpoint = CString::new(self.endpoint.as_str()).ok()?;
        let buffer_type = unsafe {
            llama_cpp_sys_2::ggml_backend_rpc_buffer_type(c_endpoint.as_ptr())
        };
        NonNull::new(buffer_type)
    }
    
    /// Query the available memory on the remote device
    ///
    /// # Returns
    ///
    /// Returns `Ok((free_memory, total_memory))` in bytes on success.
    ///
    /// # Errors
    ///
    /// Returns `MemoryQueryFailed` if the query fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_2::rpc::RpcBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let backend = RpcBackend::init("127.0.0.1:50052")?;
    /// let (free, total) = backend.get_device_memory()?;
    /// println!("Remote device memory: {}/{} bytes free", free, total);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_device_memory(&self) -> Result<(usize, usize), RpcError> {
        let c_endpoint = CString::new(self.endpoint.as_str())?;
        
        let mut free: usize = 0;
        let mut total: usize = 0;
        
        unsafe {
            llama_cpp_sys_2::ggml_backend_rpc_get_device_memory(
                c_endpoint.as_ptr(),
                &mut free,
                &mut total,
            );
        }
        
        if total == 0 {
            Err(RpcError::MemoryQueryFailed)
        } else {
            Ok((free, total))
        }
    }
    
    /// Get the endpoint this backend is connected to
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
    
    /// Get the raw backend pointer for FFI calls
    pub(crate) fn as_ptr(&self) -> NonNull<llama_cpp_sys_2::ggml_backend> {
        self.backend
    }
}

impl Drop for RpcBackend {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::ggml_backend_free(self.backend.as_ptr());
        }
    }
}

// Safety: RpcBackend can be sent between threads
unsafe impl Send for RpcBackend {}
// Safety: RpcBackend can be shared between threads (the C API is thread-safe)
unsafe impl Sync for RpcBackend {}

/// RPC device representation
///
/// This represents a remote device that can be used for distributed inference.
#[derive(Debug)]
pub struct RpcDevice {
    device: NonNull<llama_cpp_sys_2::ggml_backend_device>,
    endpoint: String,
}

impl RpcDevice {
    /// Add a new RPC device for the given endpoint
    ///
    /// This function registers a remote device that can be used for model execution.
    /// The device will be available for use with model loading and inference.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - The RPC server endpoint (e.g., "192.168.1.100:50052")
    ///
    /// # Returns
    ///
    /// Returns `Ok(RpcDevice)` on success.
    ///
    /// # Errors
    ///
    /// * `StringConversion` - Endpoint contains null bytes
    /// * `AddDeviceFailed` - Failed to add the device
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_2::rpc::RpcDevice;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Add multiple RPC devices for distributed inference
    /// let device1 = RpcDevice::add("192.168.1.100:50052")?;
    /// let device2 = RpcDevice::add("192.168.1.101:50052")?;
    /// 
    /// println!("Added devices: {} and {}", device1.endpoint(), device2.endpoint());
    /// # Ok(())
    /// # }
    /// ```
    pub fn add(endpoint: &str) -> Result<Self, RpcError> {
        let c_endpoint = CString::new(endpoint)?;
        
        let device = unsafe {
            llama_cpp_sys_2::ggml_backend_rpc_add_device(c_endpoint.as_ptr())
        };
        
        NonNull::new(device)
            .map(|ptr| Self {
                device: ptr,
                endpoint: endpoint.to_string(),
            })
            .ok_or_else(|| RpcError::AddDeviceFailed {
                endpoint: endpoint.to_string(),
            })
    }
    
    /// Get the endpoint this device is connected to
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
    
    /// Get device name
    #[must_use]
    pub fn name(&self) -> String {
        unsafe {
            let name_ptr = llama_cpp_sys_2::ggml_backend_dev_name(self.device.as_ptr());
            if name_ptr.is_null() {
                self.endpoint.clone()
            } else {
                CStr::from_ptr(name_ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }
    
    /// Get device description
    #[must_use]
    pub fn description(&self) -> String {
        unsafe {
            let desc_ptr = llama_cpp_sys_2::ggml_backend_dev_description(self.device.as_ptr());
            if desc_ptr.is_null() {
                format!("RPC device at {}", self.endpoint)
            } else {
                CStr::from_ptr(desc_ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }
}

// Safety: RpcDevice can be sent between threads
unsafe impl Send for RpcDevice {}
// Safety: RpcDevice can be shared between threads
unsafe impl Sync for RpcDevice {}

/// RPC server for hosting model inference
///
/// This allows a machine to act as an RPC server that can execute
/// tensor operations for remote clients.
pub struct RpcServer {
    backend: NonNull<llama_cpp_sys_2::ggml_backend>,
    endpoint: String,
}

impl RpcServer {
    /// Start an RPC server on the specified endpoint
    ///
    /// # Arguments
    ///
    /// * `backend` - The backend to use for serving requests
    /// * `endpoint` - The endpoint to listen on (e.g., "0.0.0.0:50052")
    /// * `cache_dir` - Optional cache directory for the server
    /// * `free_mem` - Amount of free memory to advertise (in bytes)
    /// * `total_mem` - Total memory to advertise (in bytes)
    ///
    /// # Errors
    ///
    /// Returns `ServerStartFailed` if the server cannot be started.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llama_cpp_2::rpc::RpcServer;
    /// use llama_cpp_2::llama_backend::LlamaBackend;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let backend = LlamaBackend::init()?;
    /// // Note: This would need an actual GGML backend instance
    /// // let server = RpcServer::start(
    /// //     backend_ptr,
    /// //     "0.0.0.0:50052",
    /// //     Some("/tmp/cache"),
    /// //     8_000_000_000,  // 8GB free
    /// //     16_000_000_000, // 16GB total
    /// // )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn start(
        backend: NonNull<llama_cpp_sys_2::ggml_backend>,
        endpoint: &str,
        cache_dir: Option<&str>,
        free_mem: usize,
        total_mem: usize,
    ) -> Result<Self, RpcError> {
        let c_endpoint = CString::new(endpoint)?;
        
        let c_cache_dir = cache_dir
            .map(|dir| CString::new(dir))
            .transpose()?;
        
        unsafe {
            llama_cpp_sys_2::ggml_backend_rpc_start_server(
                backend.as_ptr(),
                c_endpoint.as_ptr(),
                c_cache_dir
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                free_mem,
                total_mem,
            );
        }
        
        Ok(Self {
            backend,
            endpoint: endpoint.to_string(),
        })
    }
    
    /// Get the endpoint this server is listening on
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

impl std::fmt::Debug for RpcServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RpcServer")
            .field("endpoint", &self.endpoint)
            .finish()
    }
}

// Safety: RpcServer can be sent between threads
unsafe impl Send for RpcServer {}
// Safety: RpcServer can be shared between threads
unsafe impl Sync for RpcServer {}