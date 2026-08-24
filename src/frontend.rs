//! Pure-Rust text normalization and `GLaDOS` phoneme tokenization.
//!
//! The upstream project uses `DeepPhonemizer` for words that are absent from its
//! bundled English dictionary. Preparation converts the dictionary into a
//! small UTF-8 TSV artifact, and the runtime uses the native Burn phonemizer
//! for dictionary misses without Python.

use eyre::WrapErr;
use eyre::bail;
use std::collections::HashMap;
use std::path::Path;

/// The symbol order used by the upstream `ForwardTacotron` checkpoint.
#[expect(
    clippy::unicode_not_nfc,
    reason = "The symbol order must preserve the upstream code-point sequence."
)]
pub const GLADOS_SYMBOLS: &str = "_!'(),.:;? -iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻʘɓǀɗǃʄǂɠǁʛpbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟˈˌːˑʍwɥʜʢʡɕʑɺɧɚ˞ɫgɝ̥̩̯̃̍͡";

/// A prepared dictionary plus the upstream symbol-to-ID mapping.
#[derive(Debug)]
pub struct GladosFrontend {
    words: HashMap<String, String>,
    symbol_to_id: HashMap<char, i32>,
}

impl GladosFrontend {
    /// Load the prepared dictionary artifact.
    ///
    /// Each non-empty line has the form `word<TAB>ipa`.  This format is
    /// intentionally plain and inspectable so it can be generated during
    /// model preparation without making the runtime depend on a serializer.
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact cannot be read or contains a phoneme
    /// that is not present in the upstream symbol table.
    pub fn from_tsv(path: &Path) -> eyre::Result<Self> {
        let contents = std::fs::read_to_string(path)
            .wrap_err_with(|| format!("failed to read frontend dictionary {}", path.display()))?;
        Self::from_tsv_contents(&contents)
    }

    /// Build a frontend from an in-memory TSV artifact.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed lines or unknown IPA symbols.
    pub fn from_tsv_contents(contents: &str) -> eyre::Result<Self> {
        let symbol_to_id = GLADOS_SYMBOLS
            .chars()
            .enumerate()
            .map(|(index, symbol)| (symbol, i32::try_from(index).unwrap_or(i32::MAX)))
            .collect::<HashMap<_, _>>();
        let mut words = HashMap::new();

        for (line_number, line) in contents.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let Some((word, phonemes)) = line.split_once('\t') else {
                bail!(
                    "frontend dictionary line {} is missing a tab",
                    line_number + 1
                );
            };
            if word.is_empty() {
                bail!(
                    "frontend dictionary line {} has an empty word",
                    line_number + 1
                );
            }
            for symbol in phonemes.chars() {
                if !symbol_to_id.contains_key(&symbol) {
                    bail!(
                        "frontend dictionary line {} contains unsupported phoneme {:?}",
                        line_number + 1,
                        symbol
                    );
                }
            }
            words.insert(word.to_ascii_lowercase(), phonemes.to_string());
        }

        Ok(Self {
            words,
            symbol_to_id,
        })
    }

    /// Normalize English text and return `ForwardTacotron` token IDs.
    ///
    /// This mirrors the upstream boundary for punctuation and dictionary
    /// lookup. This strict helper rejects unknown words; use [`Self::tokenize_with`]
    /// when a native phonemizer is available.
    ///
    /// # Errors
    ///
    /// Returns an error if a word is absent from the prepared dictionary or a
    /// retained character is outside the upstream symbol table.
    pub fn tokenize(&self, text: &str) -> eyre::Result<Vec<i32>> {
        self.tokenize_with(text, |word| {
            bail!("word {:?} is not in the prepared dictionary", word)
        })
    }

    /// Normalize text and return the exact `GLaDOS` phoneme-symbol sequence that
    /// the model frontend will consume.
    ///
    /// # Errors
    ///
    /// Returns an error from the callback or when a callback result contains
    /// an unsupported symbol.
    pub fn phonemize_with<F>(&self, text: &str, mut phonemize_unknown: F) -> eyre::Result<String>
    where
        F: FnMut(&str) -> eyre::Result<String>,
    {
        let normalized = normalize_text(text);
        let mut phonemes = String::new();
        let mut word = String::new();

        for character in normalized.chars() {
            if character.is_ascii_alphanumeric() || character == '\'' {
                word.push(character.to_ascii_lowercase());
                continue;
            }

            self.flush_word_with(&mut word, &mut phonemes, &mut phonemize_unknown)?;
            if character == ' ' {
                phonemes.push(' ');
            } else if GLADOS_SYMBOLS.contains(character) {
                phonemes.push(character);
            }
        }
        self.flush_word_with(&mut word, &mut phonemes, &mut phonemize_unknown)?;

        if phonemes.is_empty() {
            bail!("text did not contain any transcribable words or punctuation");
        }
        Ok(phonemes)
    }

    /// Validate and tokenize `GLaDOS`'s IPA-like phoneme symbols directly.
    ///
    /// Unlike [`Self::tokenize`], this does not normalize text, append
    /// punctuation, or invoke the neural phonemizer. Whitespace is treated as
    /// the symbol-space separator used by the upstream model.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is empty or contains a symbol outside the
    /// upstream `GLaDOS` symbol table.
    pub fn tokenize_phonemes(&self, phonemes: &str) -> eyre::Result<Vec<i32>> {
        if phonemes.trim().is_empty() {
            bail!("phoneme input did not contain any symbols");
        }
        let mut tokens = Vec::new();
        for symbol in phonemes.chars() {
            let symbol = if symbol.is_whitespace() { ' ' } else { symbol };
            self.push_symbol(symbol, &mut tokens)?;
        }
        Ok(tokens)
    }

    /// Normalize and tokenize text, invoking `phonemize_unknown` for words
    /// absent from the prepared dictionary.
    ///
    /// # Errors
    ///
    /// Returns an error from the callback or when a callback result contains
    /// an unsupported symbol.
    pub fn tokenize_with<F>(&self, text: &str, mut phonemize_unknown: F) -> eyre::Result<Vec<i32>>
    where
        F: FnMut(&str) -> eyre::Result<String>,
    {
        let phonemes = self.phonemize_with(text, &mut phonemize_unknown)?;
        let mut tokens = Vec::new();
        for symbol in phonemes.chars() {
            self.push_symbol(symbol, &mut tokens)?;
        }
        Ok(tokens)
    }

    /// Return the number of words in the prepared dictionary.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    fn flush_word_with<F>(
        &self,
        word: &mut String,
        phonemes_output: &mut String,
        phonemize_unknown: &mut F,
    ) -> eyre::Result<()>
    where
        F: FnMut(&str) -> eyre::Result<String>,
    {
        if word.is_empty() {
            return Ok(());
        }
        let phonemes = match self.words.get(word) {
            Some(phonemes) => phonemes.clone(),
            None => phonemize_unknown(word)?,
        };
        for phoneme in phonemes.chars() {
            if !self.symbol_to_id.contains_key(&phoneme) {
                bail!(
                    "phoneme symbol {:?} is not in the GLaDOS symbol table",
                    phoneme
                );
            }
        }
        phonemes_output.push_str(&phonemes);
        word.clear();
        Ok(())
    }

    fn push_symbol(&self, symbol: char, tokens: &mut Vec<i32>) -> eyre::Result<()> {
        let Some(&token) = self.symbol_to_id.get(&symbol) else {
            bail!(
                "phoneme symbol {:?} is not in the GLaDOS symbol table",
                symbol
            );
        };
        tokens.push(token);
        Ok(())
    }
}

fn normalize_text(text: &str) -> String {
    let mut normalized = transliterate_ascii(text.trim());
    if !matches!(normalized.chars().last(), Some('.' | '?' | '!')) {
        normalized.push('.');
    }
    normalized = normalize_numbers(&normalized);
    normalized = expand_abbreviations(&normalized);
    normalized
        .chars()
        .map(|character| match character {
            '\t' | '\n' | '\r' => ' ',
            '–' | '—' => '-',
            '“' | '”' => '"',
            '‘' | '’' => '\'',
            other => other,
        })
        .filter(|character| {
            character.is_ascii_alphanumeric()
                || character.is_ascii_whitespace()
                || *character == '\''
                || "!(),.:;?/-".contains(*character)
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn transliterate_ascii(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        let replacement = match character {
            'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => {
                "a"
            }
            'Æ' => "Ae",
            'æ' => "ae",
            'Ç' => "C",
            'ç' => "c",
            'È' | 'É' | 'Ê' | 'Ë' | 'è' | 'é' | 'ê' | 'ë' => "e",
            'Ì' | 'Í' | 'Î' | 'Ï' | 'ì' | 'í' | 'î' | 'ï' => "i",
            'Ñ' => "N",
            'ñ' => "n",
            'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'ò' | 'ó' | 'ô' | 'õ' | 'ö' => "o",
            'Œ' => "Oe",
            'œ' => "oe",
            'Ù' | 'Ú' | 'Û' | 'Ü' | 'ù' | 'ú' | 'û' | 'ü' => "u",
            'Ý' | 'Ÿ' | 'ý' | 'ÿ' => "y",
            'ß' => "ss",
            '“' | '”' | '„' => "\"",
            '‘' | '’' | '‚' => "'",
            '–' | '—' | '−' => "-",
            other if other.is_ascii() => {
                output.push(other);
                continue;
            }
            _ => continue,
        };
        output.push_str(replacement);
    }
    output
}

fn normalize_numbers(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if matches!(character, '$' | '£')
            && characters.get(index + 1).is_some_and(char::is_ascii_digit)
        {
            let end = number_end(&characters, index + 1);
            let literal = characters[index + 1..end]
                .iter()
                .filter(|value| **value != ',')
                .collect::<String>();
            if character == '$' {
                output.push_str(&expand_dollars(&literal));
            } else {
                output.push_str(&integer_to_words(&literal));
                output.push_str(" pounds");
            }
            index = end;
            continue;
        }
        if character.is_ascii_digit() {
            let mut end = number_end(&characters, index);
            let mut ordinal = false;
            if characters.get(end..end + 2).is_some_and(|suffix| {
                matches!(suffix, ['s', 't'])
                    || matches!(suffix, ['n', 'd'])
                    || matches!(suffix, ['r', 'd'])
                    || matches!(suffix, ['t', 'h'])
            }) {
                ordinal = true;
                end += 2;
            }
            let literal = characters[index..number_end(&characters, index)]
                .iter()
                .filter(|value| **value != ',')
                .collect::<String>();
            let replacement = if ordinal {
                ordinal_to_words(&literal)
            } else if literal.contains('.') {
                decimal_to_words(&literal)
            } else {
                integer_to_words(&literal)
            };
            output.push_str(&replacement);
            index = end;
            continue;
        }
        output.push(character);
        index += 1;
    }
    output
}

fn number_end(characters: &[char], start: usize) -> usize {
    let mut end = start;
    while characters.get(end).is_some_and(char::is_ascii_digit) {
        end += 1;
    }
    while characters.get(end) == Some(&',')
        && characters.get(end + 1).is_some_and(char::is_ascii_digit)
    {
        end += 1;
        while characters.get(end).is_some_and(char::is_ascii_digit) {
            end += 1;
        }
    }
    if characters.get(end) == Some(&'.')
        && characters.get(end + 1).is_some_and(char::is_ascii_digit)
    {
        end += 1;
        while characters.get(end).is_some_and(char::is_ascii_digit) {
            end += 1;
        }
    }
    end
}

fn expand_dollars(literal: &str) -> String {
    let Some((dollars, cents)) = literal.split_once('.') else {
        return format!("{} dollars", integer_to_words(literal));
    };
    let dollars = dollars.trim_start_matches('0');
    let dollars = if dollars.is_empty() { "0" } else { dollars };
    if cents.parse::<u64>().is_ok_and(|value| value == 0) {
        return format!("{} dollars", integer_to_words(dollars));
    }
    format!(
        "{} dollars, {} cents",
        integer_to_words(dollars),
        integer_to_words(cents)
    )
}

fn decimal_to_words(literal: &str) -> String {
    let Some((whole, fractional)) = literal.split_once('.') else {
        return integer_to_words(literal);
    };
    let fractional = fractional
        .chars()
        .map(digit_to_word)
        .collect::<Vec<_>>()
        .join(" ");
    format!("{} point {fractional}", integer_to_words(whole))
}

fn digit_to_word(digit: char) -> &'static str {
    match digit {
        '0' => "zero",
        '1' => "one",
        '2' => "two",
        '3' => "three",
        '4' => "four",
        '5' => "five",
        '6' => "six",
        '7' => "seven",
        '8' => "eight",
        '9' => "nine",
        _ => "",
    }
}

fn ordinal_to_words(literal: &str) -> String {
    let Ok(value) = literal.parse::<u64>() else {
        return literal.to_string();
    };
    if value < 20 {
        return [
            "zeroth",
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "eleventh",
            "twelfth",
            "thirteenth",
            "fourteenth",
            "fifteenth",
            "sixteenth",
            "seventeenth",
            "eighteenth",
            "nineteenth",
        ][usize::try_from(value).expect("ordinal under twenty fits usize")]
        .to_string();
    }
    if value < 100 && value % 10 == 0 {
        return match value {
            20 => "twentieth",
            30 => "thirtieth",
            40 => "fortieth",
            50 => "fiftieth",
            60 => "sixtieth",
            70 => "seventieth",
            80 => "eightieth",
            90 => "ninetieth",
            _ => unreachable!("value is a multiple of ten below one hundred"),
        }
        .to_string();
    }
    let words = integer_to_words(literal);
    if let Some((prefix, suffix)) = words.rsplit_once('-') {
        return format!("{prefix}-{}", ordinal_to_words(suffix));
    }
    format!("{words}th")
}

fn integer_to_words(literal: &str) -> String {
    let Ok(value) = literal.parse::<u64>() else {
        return literal.to_string();
    };
    integer_value_to_words(value)
}

fn integer_value_to_words(value: u64) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    const SCALES: [(u64, &str); 4] = [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ];
    if value < 20 {
        return ONES[usize::try_from(value).expect("value under twenty fits usize")].to_string();
    }
    if value < 100 {
        let tens = value / 10;
        let remainder = value % 10;
        return if remainder == 0 {
            TENS[usize::try_from(tens).expect("tens fits usize")].to_string()
        } else {
            format!(
                "{}-{}",
                TENS[usize::try_from(tens).expect("tens fits usize")],
                ONES[usize::try_from(remainder).expect("remainder fits usize")]
            )
        };
    }
    if value < 1000 {
        let hundreds = value / 100;
        let remainder = value % 100;
        return if remainder == 0 {
            format!(
                "{} hundred",
                ONES[usize::try_from(hundreds).expect("hundreds fits usize")]
            )
        } else {
            format!(
                "{} hundred {}",
                ONES[usize::try_from(hundreds).expect("hundreds fits usize")],
                integer_value_to_words(remainder)
            )
        };
    }
    for (scale, name) in SCALES {
        if value >= scale {
            let high = value / scale;
            let remainder = value % scale;
            return if remainder == 0 {
                format!("{} {name}", integer_value_to_words(high))
            } else {
                format!(
                    "{} {name} {}",
                    integer_value_to_words(high),
                    integer_value_to_words(remainder)
                )
            };
        }
    }
    value.to_string()
}

fn expand_abbreviations(text: &str) -> String {
    const ABBREVIATIONS: [(&str, &str); 17] = [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("ft", "fort"),
    ];
    let mut output = text.to_string();
    for (abbreviation, replacement) in ABBREVIATIONS {
        output = replace_abbreviation(&output, abbreviation, replacement);
    }
    replace_abbreviation(&output, "col", "colonel")
}

fn replace_abbreviation(text: &str, abbreviation: &str, replacement: &str) -> String {
    let target = format!("{abbreviation}.");
    let lowercase = text.to_ascii_lowercase();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(relative) = lowercase[cursor..].find(&target) {
        let start = cursor + relative;
        let boundary = start == 0
            || lowercase.as_bytes()[start - 1].is_ascii_whitespace()
            || !lowercase.as_bytes()[start - 1].is_ascii_alphanumeric();
        output.push_str(&text[cursor..start]);
        if boundary {
            output.push_str(replacement);
        } else {
            output.push_str(&text[start..start + target.len()]);
        }
        cursor = start + target.len();
    }
    output.push_str(&text[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_symbol_table_has_expected_size() {
        assert_eq!(GLADOS_SYMBOLS.chars().count(), 135);
        assert_eq!(GLADOS_SYMBOLS.chars().next(), Some('_'));
    }

    #[test]
    fn dictionary_tokenization_preserves_punctuation() {
        let frontend = GladosFrontend::from_tsv_contents("hello\thɛloʊ\n").unwrap();
        let tokens = frontend.tokenize("hello!").unwrap();
        assert_eq!(
            tokens,
            vec![
                frontend.symbol_to_id[&'h'],
                frontend.symbol_to_id[&'ɛ'],
                frontend.symbol_to_id[&'l'],
                frontend.symbol_to_id[&'o'],
                frontend.symbol_to_id[&'ʊ'],
                frontend.symbol_to_id[&'!'],
            ]
        );
    }

    #[test]
    fn phonemization_exposes_the_sequence_before_tokenization() {
        let frontend = GladosFrontend::from_tsv_contents("hello\thɛloʊ\n").unwrap();
        let phonemes = frontend
            .phonemize_with("hello", |_| unreachable!("dictionary entry should be used"))
            .unwrap();
        assert_eq!(phonemes, "hɛloʊ.");
        assert_eq!(
            frontend.tokenize_phonemes(&phonemes).unwrap(),
            frontend.tokenize("hello").unwrap()
        );
    }

    #[test]
    fn unknown_words_fail_loudly() {
        let frontend = GladosFrontend::from_tsv_contents("hello\thɛloʊ\n").unwrap();
        let error = frontend.tokenize("unknown").unwrap_err();
        assert!(error.to_string().contains("not in the prepared dictionary"));
    }

    #[test]
    fn direct_phonemes_use_the_upstream_symbol_table() {
        let frontend = GladosFrontend::from_tsv_contents("").unwrap();
        let tokens = frontend.tokenize_phonemes("eɪ").unwrap();
        assert_eq!(
            tokens,
            vec![frontend.symbol_to_id[&'e'], frontend.symbol_to_id[&'ɪ']]
        );
    }

    #[test]
    fn direct_phonemes_reject_unsupported_symbols() {
        let frontend = GladosFrontend::from_tsv_contents("").unwrap();
        let error = frontend.tokenize_phonemes("🙂").unwrap_err();
        assert!(error.to_string().contains("not in the GLaDOS symbol table"));
    }

    #[test]
    fn cleaner_handles_reference_number_abbreviation_and_unicode_cases() {
        assert_eq!(normalize_text("hello!"), "hello!");
        assert_eq!(
            normalize_text("supercalifragilistic"),
            "supercalifragilistic."
        );
        assert_eq!(
            normalize_text("Mrs. 42 costs $3.50."),
            "misess forty-two costs three dollars, fifty cents."
        );
        assert_eq!(normalize_text("“Café”"), "Cafe.");
    }
}
