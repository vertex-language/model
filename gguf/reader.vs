// Little-endian reads over mapped bytes, every one bounds-checked: a
// truncated or hostile file is an error, never a read past the mapping.
package gguf

/// The longest string llama.cpp accepts (GGUF_MAX_STRING_LENGTH).
let maxString = 1 << 30

struct Cursor {
    let _p: UnsafePointer<uint8>?
    let Count: int
    var Offset: int = 0

    init(_ p: UnsafePointer<uint8>?, _ count: int) {
        self._p = p
        self.Count = count
    }

    var Remaining: int { return Count - Offset }

    func need(_ n: int, _ what: string) throws {
        if n < 0 || n > Remaining {
            throw FormatError.malformed("\(what) runs past the end of the file, at byte \(Offset)")
        }
    }

    mutating func U8(_ what: string) throws -> uint8 {
        try need(1, what)
        let v = _p![Offset]
        Offset += 1
        return v
    }

    mutating func U16(_ what: string) throws -> uint16 {
        try need(2, what)
        let p = _p!
        let v = uint16(p[Offset]) | uint16(p[Offset + 1]) << 8
        Offset += 2
        return v
    }

    mutating func U32(_ what: string) throws -> uint32 {
        try need(4, what)
        let p = _p!
        var v: uint32 = 0
        var i = 3
        while i >= 0 {
            v = v << 8 | uint32(p[Offset + i])
            i -= 1
        }
        Offset += 4
        return v
    }

    mutating func U64(_ what: string) throws -> uint64 {
        try need(8, what)
        let p = _p!
        var v: uint64 = 0
        var i = 7
        while i >= 0 {
            v = v << 8 | uint64(p[Offset + i])
            i -= 1
        }
        Offset += 8
        return v
    }

    /// Length is a u64 count that must fit what is left, at least min
    /// bytes an element.
    mutating func Length(_ min: int, _ what: string) throws -> int {
        let n = try U64(what)
        if n > uint64(Remaining / (min > 0 ? min : 1)) {
            throw FormatError.malformed("\(what) of \(n) is more than the file holds")
        }
        return int(n)
    }

    mutating func Text(_ what: string) throws -> string {
        let n = try Length(1, what)
        if n > maxString {
            throw FormatError.malformed("\(what) is \(n) bytes, over the \(maxString) allowed")
        }
        var bytes = [uint8](repeating: 0, count: n)
        let p = _p!
        var i = 0
        while i < n {
            bytes[i] = p[Offset + i]
            i += 1
        }
        Offset += n
        return string(decoding: bytes, as: UTF8.self)
    }
}
