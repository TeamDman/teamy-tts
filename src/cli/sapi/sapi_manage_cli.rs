use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use facet::Facet;
use figue as args;
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct SapiArgs {
    /// SAPI installation, diagnostics or testing operation.
    #[facet(args::subcommand)]
    pub command: SapiCommand,
}
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum SapiCommand {
    /// Register the native Windows voice; machine registration requires administrator rights.
    Install(super::install::InstallArgs),
    /// Remove the voice registration and stop its local worker.
    Uninstall(super::uninstall::UninstallArgs),
    /// Report registration, default selection, worker and model status.
    Status(super::status::StatusArgs),
    /// Render speech through SAPI, explicitly selecting Teamy for this call.
    Test(super::test::TestArgs),
    /// Shut down the resident worker.
    Stop(super::stop::StopArgs),
}
impl SapiArgs {
    pub fn invoke(self) -> eyre::Result<CliOutput> {
        match self.command {
            SapiCommand::Install(args) => args.invoke(),
            SapiCommand::Uninstall(args) => args.invoke(),
            SapiCommand::Status(args) => args.invoke(),
            SapiCommand::Test(args) => args.invoke(),
            SapiCommand::Stop(args) => args.invoke(),
        }
    }
}
