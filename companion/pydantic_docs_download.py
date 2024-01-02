import scrapy


class PySparkDocsSpider(scrapy.Spider):
    name = "docs_download"
    start_urls = [
        "https://docs.pydantic.dev/latest/",
        "https://docs.pydantic.dev/latest/concepts/models/",
        "https://docs.pydantic.dev/latest/errors/errors/",
        "https://docs.pydantic.dev/latest/api/base_model/",
        "https://docs.pydantic.dev/latest/integrations/mypy/",
    ]

    def parse(self, response):
        yield {
            "title": response.xpath("//article/h1/text()").extract_first(),
            "content": " ".join(response.xpath("//article").getall()),
            "url": response.request.url,
        }

        for next_page in response.xpath('//a[@class="md-nav__link"]/@href').getall():
            if not next_page.startswith('#'):
                yield response.follow(next_page, self.parse)
