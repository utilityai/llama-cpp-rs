use crate::model::params::unknown_kv_override_tag::UnknownKvOverrideTag;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamOverrideValue {
    Bool(bool),
    Float(f64),
    Int(i64),
    Str([std::os::raw::c_char; 128]),
}

impl ParamOverrideValue {
    #[must_use]
    pub const fn tag(&self) -> llama_cpp_bindings_sys::llama_model_kv_override_type {
        match self {
            Self::Bool(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL,
            Self::Float(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            Self::Int(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT,
            Self::Str(_) => llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR,
        }
    }

    #[must_use]
    pub const fn value(&self) -> llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
        match self {
            Self::Bool(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_bool: *value }
            }
            Self::Float(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_f64: *value }
            }
            Self::Int(value) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_i64: *value }
            }
            Self::Str(c_string) => {
                llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 { val_str: *c_string }
            }
        }
    }
}

impl TryFrom<&llama_cpp_bindings_sys::llama_model_kv_override> for ParamOverrideValue {
    type Error = UnknownKvOverrideTag;

    fn try_from(
        llama_cpp_bindings_sys::llama_model_kv_override {
            key: _,
            tag,
            __bindgen_anon_1,
        }: &llama_cpp_bindings_sys::llama_model_kv_override,
    ) -> Result<Self, Self::Error> {
        match *tag {
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT => {
                Ok(Self::Int(unsafe { __bindgen_anon_1.val_i64 }))
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT => {
                Ok(Self::Float(unsafe { __bindgen_anon_1.val_f64 }))
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL => {
                Ok(Self::Bool(unsafe { __bindgen_anon_1.val_bool }))
            }
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR => {
                Ok(Self::Str(unsafe { __bindgen_anon_1.val_str }))
            }
            unknown_tag => Err(UnknownKvOverrideTag(unknown_tag)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ParamOverrideValue;

    #[test]
    fn tag_bool() {
        let value = ParamOverrideValue::Bool(true);

        assert_eq!(
            value.tag(),
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL
        );
    }

    #[test]
    fn tag_float() {
        let value = ParamOverrideValue::Float(1.23);

        assert_eq!(
            value.tag(),
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT
        );
    }

    #[test]
    fn tag_int() {
        let value = ParamOverrideValue::Int(42);

        assert_eq!(
            value.tag(),
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT
        );
    }

    #[test]
    fn tag_str() {
        let value = ParamOverrideValue::Str([0; 128]);

        assert_eq!(
            value.tag(),
            llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR
        );
    }

    #[test]
    fn value_bool_roundtrip() {
        let value = ParamOverrideValue::Bool(true);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_bool };

        assert!(result);
    }

    #[test]
    fn value_float_roundtrip() {
        let value = ParamOverrideValue::Float(1.23);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_f64 };

        assert!((result - 1.23).abs() < f64::EPSILON);
    }

    #[test]
    fn value_int_roundtrip() {
        let value = ParamOverrideValue::Int(99);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_i64 };

        assert_eq!(result, 99);
    }

    #[test]
    fn from_ffi_override_int() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_INT,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_i64: 123,
            },
        };

        let value = ParamOverrideValue::try_from(&ffi_override).unwrap();

        assert_eq!(value, ParamOverrideValue::Int(123));
    }

    #[test]
    fn from_ffi_override_float() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_f64: 1.5,
            },
        };

        let value = ParamOverrideValue::try_from(&ffi_override).unwrap();

        assert_eq!(value, ParamOverrideValue::Float(1.5));
    }

    #[test]
    fn from_ffi_override_bool() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_BOOL,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_bool: false,
            },
        };

        let value = ParamOverrideValue::try_from(&ffi_override).unwrap();

        assert_eq!(value, ParamOverrideValue::Bool(false));
    }

    #[test]
    fn value_str_roundtrip() {
        let mut str_data = [0i8; 128];
        str_data[0] = b'h'.cast_signed();
        str_data[1] = b'i'.cast_signed();

        let value = ParamOverrideValue::Str(str_data);
        let ffi_value = value.value();
        let result = unsafe { ffi_value.val_str };

        assert_eq!(result[0], b'h'.cast_signed());
        assert_eq!(result[1], b'i'.cast_signed());
    }

    #[test]
    fn from_ffi_override_str() {
        let mut str_data = [0i8; 128];
        str_data[0] = b'a'.cast_signed();
        str_data[1] = b'b'.cast_signed();

        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: llama_cpp_bindings_sys::LLAMA_KV_OVERRIDE_TYPE_STR,
            __bindgen_anon_1: llama_cpp_bindings_sys::llama_model_kv_override__bindgen_ty_1 {
                val_str: str_data,
            },
        };

        let value = ParamOverrideValue::try_from(&ffi_override).unwrap();

        assert_eq!(value, ParamOverrideValue::Str(str_data));
    }

    #[test]
    fn unknown_tag_returns_error() {
        let ffi_override = llama_cpp_bindings_sys::llama_model_kv_override {
            key: [0; 128],
            tag: 9999,
            __bindgen_anon_1: unsafe { std::mem::zeroed() },
        };

        let result = ParamOverrideValue::try_from(&ffi_override);

        assert!(result.is_err());
    }
}
