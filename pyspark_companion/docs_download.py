import scrapy


class PySparkDocsSpider(scrapy.Spider):
    name = "docs_download"
    start_urls = [
        "https://spark.apache.org/docs/latest/api/python/index.html",
        "https://spark.apache.org/docs/latest/api/python/getting_started/index.html",
        "https://spark.apache.org/docs/latest/api/python/user_guide/index.html",
        "https://spark.apache.org/docs/latest/api/python/reference/index.html",
        "https://spark.apache.org/docs/latest/api/python/development/index.html",
        "https://spark.apache.org/docs/latest/api/python/migration_guide/index.html"
    ]

    def parse(self, response):
        yield {
            "title": response.xpath(
                '//div[@class="section"]/h1//text()'
            ).extract_first(),
            "content": " ".join(
                response.xpath('//div[@class="section"]/p//text()').getall()
            )
            + " "
            + "".join(response.xpath('//div[@class="section"]/dl//text()').getall()),
            "url": response.request.url,
        }

        for next_page in response.xpath('//a[@class="reference internal"]/@href'):
            yield response.follow(next_page, self.parse)
