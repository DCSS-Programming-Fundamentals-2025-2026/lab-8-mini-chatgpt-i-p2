public class DefaultFileSystem : IFileSystem
{
    public string ReadAllText(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            throw new FileNotFoundException("Не знайдено шлях");
        }

        return File.ReadAllText(path);
    }

    public bool Exists(string path)
    {
        return File.Exists(path);
    }
}