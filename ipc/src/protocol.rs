use serde::{Deserialize, Serialize};
use std::io::{self, ErrorKind};

pub const VERSION: u16 = 1;
pub const HEADER: usize = 20;
pub const MAX_PAYLOAD: usize = 65_536;
pub const MAX_BUFFER: usize = MAX_PAYLOAD * 48;
pub const MAX_TEXT_CHARS: usize = 280;
pub const MAX_AUDIO_BYTES: usize = 22_050 * 2 * 60;
pub const HELLO: u16 = 1;
pub const STATUS: u16 = 2;
pub const SYNTHESIZE: u16 = 3;
pub const AUDIO: u16 = 4;
pub const DONE: u16 = 5;
pub const CANCEL: u16 = 6;
pub const ERROR: u16 = 7;
pub const PING: u16 = 8;
pub const STOP: u16 = 9;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub text: String,
    pub voice: String,
    pub rate: i32,
}
impl Request {
    pub fn validate(&self) -> io::Result<()> {
        if self.text.trim().is_empty()
            || self.text.chars().count() > MAX_TEXT_CHARS
            || !matches!(self.voice.as_str(), "p1" | "p2")
            || !(-10..=10).contains(&self.rate)
        {
            return Err(invalid("invalid text, voice or speech rate"));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Status {
    pub state: String,
    pub pid: u32,
    pub version: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub kind: u16,
    pub id: u64,
    pub payload: Vec<u8>,
}
impl Frame {
    pub fn empty(kind: u16, id: u64) -> Self {
        Self {
            kind,
            id,
            payload: Vec::new(),
        }
    }
    pub fn json(kind: u16, id: u64, value: &impl Serialize) -> io::Result<Self> {
        Ok(Self {
            kind,
            id,
            payload: serde_json::to_vec(value)?,
        })
    }
    pub fn error(id: u64, value: &str) -> Self {
        Self {
            kind: ERROR,
            id,
            payload: value.chars().take(2048).collect::<String>().into_bytes(),
        }
    }
    pub fn parse<T: for<'de> Deserialize<'de>>(&self) -> io::Result<T> {
        Ok(serde_json::from_slice(&self.payload)?)
    }
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        if self.payload.len() > MAX_PAYLOAD {
            return Err(invalid("oversized frame"));
        }
        let mut out = Vec::with_capacity(HEADER + self.payload.len());
        out.extend_from_slice(b"TTSP");
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.kind.to_le_bytes());
        out.extend_from_slice(&self.id.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.payload);
        Ok(out)
    }
}

pub fn invalid(message: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

pub fn decode(buffer: &mut Vec<u8>) -> io::Result<Option<Frame>> {
    if buffer.len() < HEADER {
        return Ok(None);
    }
    if &buffer[..4] != b"TTSP" || u16::from_le_bytes(buffer[4..6].try_into().unwrap()) != VERSION {
        return Err(invalid("incompatible Teamy TTS protocol"));
    }
    let len = u32::from_le_bytes(buffer[16..20].try_into().unwrap()) as usize;
    if len > MAX_PAYLOAD {
        return Err(invalid("oversized frame"));
    }
    if buffer.len() < HEADER + len {
        return Ok(None);
    }
    let frame = Frame {
        kind: u16::from_le_bytes(buffer[6..8].try_into().unwrap()),
        id: u64::from_le_bytes(buffer[8..16].try_into().unwrap()),
        payload: buffer[HEADER..HEADER + len].to_vec(),
    };
    buffer.drain(..HEADER + len);
    Ok(Some(frame))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_and_concatenated_frames() {
        let f = Frame::json(
            SYNTHESIZE,
            7,
            &Request {
                text: "Hi 😀".into(),
                voice: "p2".into(),
                rate: 0,
            },
        )
        .unwrap();
        let bytes = f.encode().unwrap();
        let mut buffer = Vec::new();
        for byte in &bytes[..bytes.len() - 1] {
            buffer.push(*byte);
            assert!(decode(&mut buffer).unwrap().is_none());
        }
        buffer.push(bytes[bytes.len() - 1]);
        buffer.extend(Frame::empty(CANCEL, 7).encode().unwrap());
        assert_eq!(decode(&mut buffer).unwrap().unwrap(), f);
        assert_eq!(decode(&mut buffer).unwrap().unwrap().kind, CANCEL);
        assert!(buffer.is_empty());
    }
    #[test]
    fn reject_size_and_version_before_payload_arrives() {
        let mut bytes = Frame::empty(HELLO, 0).encode().unwrap();
        bytes[16..20].copy_from_slice(&((MAX_PAYLOAD + 1) as u32).to_le_bytes());
        assert!(decode(&mut bytes).is_err());
        bytes[16..20].copy_from_slice(&0u32.to_le_bytes());
        bytes[4] = 99;
        assert!(decode(&mut bytes).is_err());
    }
}
