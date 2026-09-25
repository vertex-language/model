// A metadata value, and the error a malformed file is.
package gguf

/// Value is one metadata value: a scalar, a string, or an array of either.
/// Arrays do not nest, as llama.cpp's reader requires.
public enum Value {
    case U8(uint8)
    case I8(int8)
    case U16(uint16)
    case I16(int16)
    case U32(uint32)
    case I32(int32)
    case F32(float32)
    case Bool(bool)
    case Text(string)
    case U64(uint64)
    case I64(int64)
    case F64(float64)
    case Array(ValueType, [Value])

    /// Type is the value's GGUF type.
    public var Type: ValueType {
        switch self {
        case .U8: return .U8
        case .I8: return .I8
        case .U16: return .U16
        case .I16: return .I16
        case .U32: return .U32
        case .I32: return .I32
        case .F32: return .F32
        case .Bool: return .Bool
        case .Text: return .Text
        case .U64: return .U64
        case .I64: return .I64
        case .F64: return .F64
        case .Array: return .Array
        }
    }

    /// AsInteger is an integer value of any width, or nil for anything
    /// else (a u64 above int's range too).
    public var AsInteger: int? {
        switch self {
        case .U8(let v): return int(v)
        case .I8(let v): return int(v)
        case .U16(let v): return int(v)
        case .I16(let v): return int(v)
        case .U32(let v): return int(v)
        case .I32(let v): return int(v)
        case .U64(let v): return v <= uint64(int.max) ? int(v) : nil
        case .I64(let v): return int(v)
        default: return nil
        }
    }

    /// AsNumber is a float value, or an integer's, as a float64.
    public var AsNumber: float64? {
        switch self {
        case .F32(let v): return float64(v)
        case .F64(let v): return v
        default:
            if let n = AsInteger { return float64(n) }
            return nil
        }
    }

    /// AsText is a string value, or nil.
    public var AsText: string? {
        if case .Text(let s) = self { return s }
        return nil
    }

    /// AsFlag is a bool value, or nil.
    public var AsFlag: bool? {
        if case .Bool(let b) = self { return b }
        return nil
    }
}

/// FormatError is a file that is not well-formed GGUF, and says where.
public enum FormatError: Error {
    case malformed(string)

    public var Message: string {
        switch self {
        case .malformed(let why): return "gguf: " + why
        }
    }
}
