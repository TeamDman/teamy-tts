//! Bounded sentence units shared by synthesis and SAPI sentence skipping.
pub const MAX_TEXT_UTF16: usize = 16_384;
pub const MAX_UNIT_CHARS: usize = 240;

#[derive(Debug, Clone)]
pub struct Sentence {
    pub text: String,
    pub offset: u32,
    pub len: u32,
}

pub fn sentences(text: &str, source_offset: u32) -> Vec<Sentence> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut units = source_offset;
    let mut iter = text.char_indices().peekable();
    while let Some((index, ch)) = iter.next() {
        let next_is_space = iter.peek().is_none_or(|(_, c)| c.is_whitespace());
        let word = text[start..index].split_whitespace().last().unwrap_or("");
        let abbreviation = matches!(
            word.to_ascii_lowercase().as_str(),
            "mr" | "mrs" | "ms" | "dr" | "prof" | "sr" | "jr" | "st" | "e.g" | "i.e"
        ) || (word.chars().count() == 1
            && word.chars().all(char::is_alphabetic));
        let boundary = ch == '\n'
            || (matches!(ch, '.' | '!' | '?') && next_is_space && !(ch == '.' && abbreviation));
        if boundary {
            let end = index + ch.len_utf8();
            push(&mut result, &text[start..end], units);
            units += text[start..end].encode_utf16().count() as u32;
            start = end;
        }
    }
    push(&mut result, &text[start..], units);
    result
}

/// Inference chunks are not counted as sentences by SAPI Skip.
pub fn chunks(text: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut count = 0;
    for (i, ch) in text.char_indices() {
        count += 1;
        if (count >= MAX_UNIT_CHARS && ch.is_whitespace()) || count >= MAX_UNIT_CHARS + 40 {
            let end = i + ch.len_utf8();
            result.push(&text[start..end]);
            start = end;
            count = 0;
        }
    }
    if start < text.len() {
        result.push(&text[start..]);
    }
    result
}

fn push(result: &mut Vec<Sentence>, text: &str, offset: u32) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    let prefix = &text[..text.len() - text.trim_start().len()];
    result.push(Sentence {
        text: trimmed.to_owned(),
        offset: offset + prefix.encode_utf16().count() as u32,
        len: trimmed.encode_utf16().count() as u32,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utf16_offsets_and_bounded_unicode() {
        let s = sentences("😀 Hi.  Bye!", 10);
        assert_eq!((s[0].offset, s[0].len), (10, 6));
        assert_eq!((s[1].offset, s[1].len), (18, 4));
        let long = "😀".repeat(1000);
        let s = sentences(&long, 0);
        assert_eq!(s.iter().map(|s| s.len).sum::<u32>(), 2000);
        assert_eq!(s.len(), 1);
        let chunks = chunks(&long);
        assert_eq!(chunks.concat(), long);
        assert!(
            chunks
                .iter()
                .all(|s| s.chars().count() <= MAX_UNIT_CHARS + 40)
        );
    }
    #[test]
    fn names_decimals_and_newlines() {
        let s = sentences("Dr. Ada has 1.5 hearts.\nNext menu\nLast menu", 0);
        assert_eq!(
            s.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
            ["Dr. Ada has 1.5 hearts.", "Next menu", "Last menu"]
        );
    }
}
