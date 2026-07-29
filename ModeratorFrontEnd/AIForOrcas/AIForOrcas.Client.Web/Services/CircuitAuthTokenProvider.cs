namespace AIForOrcas.Client.Web.Services;

public class CircuitAuthTokenProvider : AIForOrcas.Client.BL.Services.IAuthTokenProvider
{
    private readonly ITokenStore _tokenStore;
    private readonly CircuitHandlerService _circuit;

    public CircuitAuthTokenProvider(ITokenStore tokenStore, CircuitHandlerService circuit)
    {
        _tokenStore = tokenStore;
        _circuit = circuit;
    }

    public string GetToken()
    {
        var circuitId = _circuit.CircuitId;
        return string.IsNullOrWhiteSpace(circuitId) ? null : _tokenStore.GetToken(circuitId);
    }
}
