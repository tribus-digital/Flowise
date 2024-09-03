import { TextSplitter } from 'langchain/text_splitter'
import { omit } from 'lodash'
import { test } from 'linkifyjs'
import { parse } from 'css-what'
import { load, SelectorType } from 'cheerio'
import { webCrawl, xmlScrape } from '../../../src'

import { ICommonObject, IDocument, INode, INodeData, INodeParams } from '../../../src/Interface'
import { integer } from '@opensearch-project/opensearch/api/types'

class Cheerio_DocumentLoaders implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]

    constructor() {
        this.label = 'Cheerio Web Scraper'
        this.name = 'cheerioWebScraper'
        this.version = 1.3
        this.type = 'Document'
        this.icon = 'cheerio.svg'
        this.category = 'Document Loaders'
        this.description = `Load data from webpages`
        this.baseClasses = [this.type]
        this.inputs = [
            {
                label: 'URL',
                name: 'url',
                type: 'string'
            },
            {
                label: 'Text Splitter',
                name: 'textSplitter',
                type: 'TextSplitter',
                optional: true
            },
            {
                label: 'Get Relative Links Method',
                name: 'relativeLinksMethod',
                type: 'options',
                description: 'Select a method to retrieve relative links',
                options: [
                    {
                        label: 'Web Crawl',
                        name: 'webCrawl',
                        description: 'Crawl relative links from HTML URL'
                    },
                    {
                        label: 'Scrape XML Sitemap',
                        name: 'scrapeXMLSitemap',
                        description: 'Scrape relative links from XML sitemap URL'
                    }
                ],
                default: 'webCrawl',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Get Relative Links Limit',
                name: 'limit',
                type: 'number',
                optional: true,
                default: '10',
                additionalParams: true,
                description:
                    'Only used when "Get Relative Links Method" is selected. Set 0 to retrieve all relative links, default limit is 10.',
                warning: `Retrieving all links might take long time, and all links will be upserted again if the flow's state changed (eg: different URL, chunk size, etc)`
            },
            {
                label: 'Selector (CSS)',
                name: 'selector',
                type: 'string',
                description: 'Specify a CSS selector to select the content to be extracted',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Additional Metadata',
                name: 'metadata',
                type: 'json',
                description: 'Additional metadata to be added to the extracted documents',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Omit Metadata Keys',
                name: 'omitMetadataKeys',
                type: 'string',
                rows: 4,
                description:
                    'Each document loader comes with a default set of metadata keys that are extracted from the document. You can use this field to omit some of the default metadata keys. The value should be a list of keys, seperated by comma. Use * to omit all metadata keys execept the ones you specify in the Additional Metadata field',
                placeholder: 'key1, key2, key3.nestedKey1',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Reject Error Responses',
                description: 'Reject documents with error status codes (4xx, 5xx) from the output',
                name: 'rejectErrorStatuses',
                type: 'boolean',
                default: true,
                optional: true,
                additionalParams: true
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const textSplitter = nodeData.inputs?.textSplitter as TextSplitter
        const metadata = nodeData.inputs?.metadata
        const relativeLinksMethod = nodeData.inputs?.relativeLinksMethod as string
        const selectedLinks = nodeData.inputs?.selectedLinks as string[]
        const rejectErrorStatuses = nodeData.inputs?.rejectErrorStatuses as boolean
        let limit = parseInt(nodeData.inputs?.limit as string)

        const _omitMetadataKeys = nodeData.inputs?.omitMetadataKeys as string

        let omitMetadataKeys: string[] = []
        if (_omitMetadataKeys) {
            omitMetadataKeys = _omitMetadataKeys.split(',').map((key) => key.trim())
        }

        let url = nodeData.inputs?.url as string
        url = url.trim()
        if (!test(url)) {
            throw new Error('Invalid URL')
        }

        const isDebug = process.env.DEBUG === 'true'
        let errorURLs: Map<string, integer> = new Map()

        const selector: SelectorType = nodeData.inputs?.selector as SelectorType
        if (selector) parse(selector) // will throw error if invalid

        async function cheerioLoader(url: string): Promise<any> {
            try {
                if (url.endsWith('.pdf')) {
                    if (isDebug) options.logger.info(`Cheerio does not support PDF files: ${url}`)
                    return [] as IDocument[]
                }

                if (isDebug) options.logger.info(`Fetching content of: ${url}`)

                const response = await fetch(url)

                if (!response.ok) {
                    errorURLs.set(url, response.status as integer)
                    if (isDebug) options.logger.error(`HTTP error - status: ${response.status}`)
                    if (rejectErrorStatuses) return [] as IDocument[]
                }

                if (isDebug) options.logger.info(`Response status code: ${response.status}`)

                // Read the response body as text
                const data: string = await response.text()

                // Load the HTML content into Cheerio
                const $ = load(data)

                // get the content of the selector if provided, otherwise get the entire body text
                const content = selector ? $(selector).text() : $('body').text()

                // select the meta tag with property="og:title" and get the content attribute
                const ogTitle = $("meta[property='og:title']")?.attr('content')
                // if og:title is not present, use the page title node
                const title = ogTitle ? ogTitle : $('title').text()

                if (isDebug) options.logger.info(`title: ${title}, url: ${url}`)
                // TODO: think about allowing user to specify the page meta(data) key:value pairs to extract (or omit?)

                // create the initial document object
                let docs: IDocument[] = [
                    {
                        pageContent: content,
                        metadata: {
                            source: url,
                            title: title
                        }
                    }
                ]

                // split into chunks if a text splitter is specified
                if (textSplitter) docs = await textSplitter.splitDocuments(docs)

                return docs
            } catch (err) {
                if (isDebug) options.logger.error(`error in CheerioWebBaseLoader: ${err.message}, on page: ${url}`)
                return []
            }
        }

        let docs: IDocument[] = []

        if (relativeLinksMethod) {
            if (isDebug) options.logger.info(`Start ${relativeLinksMethod}`)
            // if limit is 0 we don't want it to default to 10 so we check explicitly for null or undefined
            // so when limit is 0 we can fetch all the links
            if (limit === null || limit === undefined) limit = 10
            else if (limit < 0) throw new Error('Limit cannot be less than 0')
            const pages: string[] =
                selectedLinks && selectedLinks.length > 0
                    ? selectedLinks.slice(0, limit === 0 ? undefined : limit)
                    : relativeLinksMethod === 'webCrawl'
                    ? await webCrawl(url, limit)
                    : await xmlScrape(url, limit)
            if (isDebug) options.logger.info(`pages: ${JSON.stringify(pages)}, length: ${pages.length}`)
            if (!pages || pages.length === 0) throw new Error('No relative links found')
            for (const page of pages) {
                docs.push(...(await cheerioLoader(page)))
            }
            if (isDebug) options.logger.info(`Finish ${relativeLinksMethod}`)
        } else if (selectedLinks && selectedLinks.length > 0) {
            if (isDebug) options.logger.info(`pages: ${JSON.stringify(selectedLinks)}, length: ${selectedLinks.length}`)
            for (const page of selectedLinks.slice(0, limit)) {
                docs.push(...(await cheerioLoader(page)))
            }
        } else {
            docs = await cheerioLoader(url)
        }

        if (metadata) {
            const parsedMetadata = typeof metadata === 'object' ? metadata : JSON.parse(metadata)
            docs = docs.map((doc) => ({
                ...doc,
                metadata:
                    _omitMetadataKeys === '*'
                        ? {
                              ...parsedMetadata
                          }
                        : omit(
                              {
                                  ...doc.metadata,
                                  ...parsedMetadata
                              },
                              omitMetadataKeys
                          )
            }))
        } else {
            docs = docs.map((doc) => ({
                ...doc,
                metadata:
                    _omitMetadataKeys === '*'
                        ? {}
                        : omit(
                              {
                                  ...doc.metadata
                              },
                              omitMetadataKeys
                          )
            }))
        }

        if (isDebug) {
            options.logger.info(`Scrape completed`)
            options.logger.info(`Total error responses: ${errorURLs.size}`)
            for (const item of errorURLs) options.logger.info(`URL: ${item[0]}, status: ${item[1]}`)
        }

        return docs
    }
}

module.exports = { nodeClass: Cheerio_DocumentLoaders }
