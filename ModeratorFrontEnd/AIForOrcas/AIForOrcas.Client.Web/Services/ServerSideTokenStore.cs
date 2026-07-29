using System.Collections.Concurrent;

namespace AIForOrcas.Client.Web.Services;

public interface ITokenStore
{
    void SetToken(string circuitId, string token);
    string GetToken(string circuitId);
    void RemoveToken(string circuitId);
}

public class ServerSideTokenStore : ITokenStore
{
    private static readonly TimeSpan TokenTtl = TimeSpan.FromMinutes(30);
    private readonly ConcurrentDictionary<string, TokenEntry> _tokens = new();

    public void SetToken(string circuitId, string token)
    {
        CleanupExpiredTokens();
        _tokens[circuitId] = new TokenEntry(token, DateTimeOffset.UtcNow.Add(TokenTtl));
    }

    public string GetToken(string circuitId)
    {
        CleanupExpiredTokens();
        if (!_tokens.TryGetValue(circuitId, out var tokenEntry))
        {
            return null;
        }

        if (tokenEntry.ExpiresAtUtc <= DateTimeOffset.UtcNow)
        {
            _tokens.TryRemove(circuitId, out _);
            return null;
        }

        _tokens[circuitId] = tokenEntry with { ExpiresAtUtc = DateTimeOffset.UtcNow.Add(TokenTtl) };
        return tokenEntry.Token;
    }

    public void RemoveToken(string circuitId)
    {
        _tokens.TryRemove(circuitId, out _);
    }

    private void CleanupExpiredTokens()
    {
        var now = DateTimeOffset.UtcNow;
        foreach (var token in _tokens.ToArray())
        {
            if (token.Value.ExpiresAtUtc <= now)
            {
                _tokens.TryRemove(token.Key, out _);
            }
        }
    }

    private readonly record struct TokenEntry(string Token, DateTimeOffset ExpiresAtUtc);
}
