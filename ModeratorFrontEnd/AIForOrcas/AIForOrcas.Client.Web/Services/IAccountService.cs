namespace AIForOrcas.Client.Web.Services;

public interface IAccountService
{
    Task Login();
    Task Logout();
    string GetToken();
    Task<string> GetDisplayname();
    Task<string> GetUsername();
}
