using Microsoft.AspNetCore.Components.Server.Circuits;

namespace AIForOrcas.Client.Web.Services;

public class CircuitHandlerService : CircuitHandler
{
    private readonly ITokenStore _tokenStore;
    private readonly BlazoradeMsalService _msalService;
    private readonly AppSettings _appSettings;
    private readonly Microsoft.Extensions.Logging.ILogger<CircuitHandlerService> _logger;

    public CircuitHandlerService(
        ITokenStore tokenStore,
        BlazoradeMsalService msalService,
        AppSettings appSettings,
        Microsoft.Extensions.Logging.ILogger<CircuitHandlerService> logger)
    {
        _tokenStore = tokenStore;
        _msalService = msalService;
        _appSettings = appSettings;
        _logger = logger;
    }

    public string CircuitId { get; private set; }

    public override async Task OnCircuitOpenedAsync(Circuit circuit, CancellationToken cancellationToken)
    {
        CircuitId = circuit.Id;
        Microsoft.Extensions.Logging.LoggerExtensions.LogDebug(_logger, "Circuit opened: {CircuitId}", CircuitId);

        // Attempt silent MSAL re-authentication so the user is transparently
        // re-authenticated after a browser page refresh without a login prompt.
        await TryAcquireTokenSilentlyAsync(circuit.Id);
    }

    public override Task OnCircuitClosedAsync(Circuit circuit, CancellationToken cancellationToken)
    {
        Microsoft.Extensions.Logging.LoggerExtensions.LogDebug(_logger, "Circuit closed: {CircuitId}", circuit.Id);
        _tokenStore.RemoveToken(circuit.Id);
        CircuitId = null;
        return Task.CompletedTask;
    }

    private async Task TryAcquireTokenSilentlyAsync(string circuitId)
    {
        var clientId = _appSettings?.AzureAd?.ClientId;
        var defaultScope = _appSettings?.AzureAd?.DefaultScope;

        if (string.IsNullOrWhiteSpace(clientId) || string.IsNullOrWhiteSpace(defaultScope))
        {
            Microsoft.Extensions.Logging.LoggerExtensions.LogWarning(_logger, "Azure AD ClientId or DefaultScope is not configured; skipping silent token acquisition for circuit {CircuitId}", circuitId);
            return;
        }

        try
        {
            var scopes = new string[] { $"api://{clientId}/{defaultScope}" };
            var token = await _msalService.AcquireTokenSilentAsync(scopes: scopes, fallbackToDefaultLoginHint: true);

            if (token?.AccessToken != null)
            {
                _tokenStore.SetToken(circuitId, token.AccessToken);
                Microsoft.Extensions.Logging.LoggerExtensions.LogInformation(_logger, "Silently re-authenticated user on circuit open: {CircuitId}", circuitId);
            }
        }
        catch (Exception ex)
        {
            // Silent authentication failure is expected when no MSAL-cached session exists.
            // The user will need to log in explicitly.
            Microsoft.Extensions.Logging.LoggerExtensions.LogDebug(_logger, ex, "Silent token acquisition failed for circuit {CircuitId}; user will log in explicitly", circuitId);
        }
    }
}
