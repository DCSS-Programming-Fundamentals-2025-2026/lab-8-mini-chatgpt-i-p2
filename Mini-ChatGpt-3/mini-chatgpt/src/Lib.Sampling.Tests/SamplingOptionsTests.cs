using MiniChatGPT.Sampling.Options;

namespace Lib.Sampling.Tests.Options
{
    public class SamplingOptionsTests
    {
        [Test]
        public void Constructor_ShouldSetDefaultValues()
        {
            var options = new SamplingOptions();

            Assert.That(options.Temperature, Is.EqualTo(1.0f), "Дефолтна температура має бути 1.0f");
            Assert.That(options.TopK, Is.EqualTo(10), "Дефолтне значення TopK має бути 10");
            Assert.That(options.Seed, Is.Null, "Дефолтне значення Seed має бути null");
        }

        [Test]
        public void Properties_CanBeSetAndRetrieved()
        {
            var options = new SamplingOptions();
            float expectedTemperature = 0.7f;
            int expectedTopK = 50;
            int expectedSeed = 12345;

            options.Temperature = expectedTemperature;
            options.TopK = expectedTopK;
            options.Seed = expectedSeed;

            Assert.That(options.Temperature, Is.EqualTo(expectedTemperature));
            Assert.That(options.TopK, Is.EqualTo(expectedTopK));
            Assert.That(options.Seed, Is.EqualTo(expectedSeed));
        }
    }
}