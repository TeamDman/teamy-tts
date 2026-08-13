use arbitrary::Arbitrary;
use eyre::Context;
use facet::Facet;
use facet_pretty::ColorMode;
use facet_pretty::PrettyPrinter;
use std::io::IsTerminal;
use std::io::Write;
use std::io::{self};

#[derive(Arbitrary, Facet, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[facet(rename_all = "kebab-case")]
#[repr(u8)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Csv,
}

pub struct CliOutput(Option<Box<dyn CliOutputValue>>);

impl core::fmt::Debug for CliOutput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CliOutput")
            .field("has_value", &self.0.is_some())
            .finish()
    }
}

trait CliOutputValue {
    fn render(
        &self,
        format: OutputFormat,
        stdout_is_terminal: bool,
    ) -> eyre::Result<Option<String>>;
}

struct FacetCliOutput<T> {
    value: T,
    csv: Option<String>,
}

impl CliOutput {
    #[must_use]
    pub const fn none() -> Self {
        Self(None)
    }

    #[must_use]
    pub fn facet<T>(value: T) -> Self
    where
        T: Facet<'static> + 'static,
    {
        Self(Some(Box::new(FacetCliOutput { value, csv: None })))
    }

    /// Return a typed output value with an explicit flat CSV projection.
    ///
    /// Facet CSV intentionally rejects nested structures and sequences. This
    /// keeps those limitations in the shared output layer while allowing a
    /// command such as `doctor` to expose one row per diagnostic check without
    /// changing its canonical text or JSON report.
    #[must_use]
    pub fn facet_with_csv<T>(value: T, csv: String) -> Self
    where
        T: Facet<'static> + 'static,
    {
        Self(Some(Box::new(FacetCliOutput {
            value,
            csv: Some(csv),
        })))
    }

    /// # Errors
    ///
    /// This function will return an error if the selected output format cannot be rendered
    /// or if the rendered output cannot be written to stdout.
    pub fn emit(self, requested_format: Option<OutputFormat>) -> eyre::Result<()> {
        let Some(output) = self.0 else {
            return Ok(());
        };

        let stdout_is_terminal = io::stdout().is_terminal();
        let format = requested_format.unwrap_or(if stdout_is_terminal {
            OutputFormat::Text
        } else {
            OutputFormat::Json
        });
        let Some(rendered) = output.render(format, stdout_is_terminal)? else {
            return Ok(());
        };

        let mut stdout = io::stdout().lock();
        stdout
            .write_all(rendered.as_bytes())
            .wrap_err("failed to write command output")?;
        if !rendered.ends_with('\n') {
            stdout
                .write_all(b"\n")
                .wrap_err("failed to terminate command output")?;
        }
        stdout.flush().wrap_err("failed to flush command output")?;
        Ok(())
    }
}

impl Default for CliOutput {
    fn default() -> Self {
        Self::none()
    }
}

impl<T> CliOutputValue for FacetCliOutput<T>
where
    T: Facet<'static> + 'static,
{
    fn render(
        &self,
        format: OutputFormat,
        stdout_is_terminal: bool,
    ) -> eyre::Result<Option<String>> {
        let rendered = match format {
            OutputFormat::Text => PrettyPrinter::new()
                .with_colors(if stdout_is_terminal {
                    ColorMode::Always
                } else {
                    ColorMode::Never
                })
                .format(&self.value),
            OutputFormat::Json => facet_json::to_string_pretty(&self.value)
                .wrap_err("failed to serialize command output as JSON")?,
            OutputFormat::Csv => match self.csv.as_deref() {
                Some(csv) => csv.to_string(),
                None => facet_csv::to_string(&self.value)
                    .wrap_err("failed to serialize command output as CSV")?,
            },
        };
        Ok(Some(rendered))
    }
}
