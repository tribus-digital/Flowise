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
        this.version = 1.4
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
                default: 10,
                additionalParams: true,
                description:
                    'Only used when "Get Relative Links Method" is selected. Set 0 to retrieve all relative links, default limit is 10.',
                warning: `Retrieving all links might take long time, and all links will be upserted again if the flow's state changed (eg: different URL, chunk size, etc)`
            },
            {
                label: 'Crawl subdomains',
                name: 'allowSubdomains',
                type: 'boolean',
                optional: true,
                default: true,
                additionalParams: true,
                description: 'Allow crawling of relative links on subdomains when using the "Web Crawl" method'
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
                name: 'rejectErrorResponses',
                type: 'boolean',
                default: true,
                optional: true,
                additionalParams: true
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const isDebug = process.env.DEBUG === 'true'

        const textSplitter = nodeData.inputs?.textSplitter as TextSplitter
        const metadata = nodeData.inputs?.metadata
        const relativeLinksMethod = nodeData.inputs?.relativeLinksMethod as string
        const selectedLinks = nodeData.inputs?.selectedLinks as string[]
        const rejectErrorResponses = nodeData.inputs?.rejectErrorResponses as boolean
        const allowSubdomains = nodeData.inputs?.allowSubdomains as boolean

        let limit = parseInt(`${nodeData.inputs?.limit}`, 10)
        if (isNaN(limit) || limit < 0) limit = 10

        if (isDebug) {
            // log input settings
            options.logger.info(`textSplitter: ${textSplitter}`)
            options.logger.info(`metadata: ${metadata}`)
            options.logger.info(`relativeLinksMethod: ${relativeLinksMethod}`)
            options.logger.info(`selectedLinks: ${selectedLinks}`)
            options.logger.info(`rejectErrorResponses: ${rejectErrorResponses}`)
            options.logger.info(`allowSubdomains: ${allowSubdomains}`)
            options.logger.info(`limit: ${limit}`)
        }

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

        let errorURLs: Map<string, integer> = new Map()
        let processedURLs: Set<string> = new Set()

        const selector: SelectorType = nodeData.inputs?.selector as SelectorType
        if (selector) parse(selector) // will throw error if invalid

        async function cheerioLoader(url: string): Promise<IDocument[]> {
            try {
                if (processedURLs.has(url)) {
                    if (isDebug) options.logger.info(`URL already processed: ${url}`)
                    return [] as IDocument[]
                }
                processedURLs.add(url)

                if (url.endsWith('.pdf')) {
                    if (isDebug) options.logger.info(`Cheerio does not support PDF files: ${url}`)
                    return [] as IDocument[]
                }

                if (isDebug) options.logger.info(`Fetching content of: ${url}`)

                const response = await fetch(url)

                if (!response.ok) {
                    errorURLs.set(url, response.status as integer)
                    if (isDebug) options.logger.error(`HTTP error - status: ${response.status}`)
                    if (rejectErrorResponses) return [] as IDocument[]
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
                if (isDebug) options.logger.error(`error in cheerioLoader: ${err.message}, on page: ${url}`)
                return [] as IDocument[]
            }
        }

        let docs: IDocument[] = []

        if (relativeLinksMethod) {
            if (isDebug) options.logger.info(`Start ${relativeLinksMethod}`)

            // Determine pages to process based on selectedLinks and relativeLinksMethod.
            let pages: string[]
            if (selectedLinks && selectedLinks.length > 0) {
                // If specific links are selected, use them up to the specified limit.
                pages = selectedLinks.slice(0, limit === 0 ? undefined : limit)
            } else if (relativeLinksMethod === 'webCrawl') {
                // If 'webCrawl' method is selected, fetch pages using web crawling.
                pages = await webCrawl(url, limit, allowSubdomains)
            } else {
                // Otherwise, fetch pages using XML scraping.
                pages = await xmlScrape(url, limit)
            }

            if (isDebug) options.logger.info(`pages: ${JSON.stringify(pages)}, length: ${pages.length}`)

            // If no pages were found, throw an error.
            if (!pages || pages.length === 0) {
                throw new Error('No relative links found')
            }

            // Process each page to extract content using cheerioLoader.
            for (const page of pages) {
                const loadedDocs = await cheerioLoader(page)
                docs.push(...loadedDocs)
            }

            if (isDebug) options.logger.info(`Finish ${relativeLinksMethod}`)
        } else if (selectedLinks && selectedLinks.length > 0) {
            // If no relativeLinksMethod but selectedLinks are provided, process them.
            if (isDebug) options.logger.info(`pages: ${JSON.stringify(selectedLinks)}, length: ${selectedLinks.length}`)

            // Process each selected link up to the specified limit.
            for (const page of selectedLinks.slice(0, limit)) {
                const loadedDocs = await cheerioLoader(page)
                docs.push(...loadedDocs)
            }
        } else {
            // If neither relativeLinksMethod nor selectedLinks are provided, load from the base URL.
            docs = await cheerioLoader(url)
        }

        // If metadata is provided, update the metadata for each document.
        if (metadata) {
            // Parse metadata if it's a string; otherwise, use it directly.
            const parsedMetadata = typeof metadata === 'object' ? metadata : JSON.parse(metadata)

            // Update each document's metadata.
            docs = docs.map((doc) => {
                const combinedMetadata = {
                    ...doc.metadata,
                    ...parsedMetadata
                }

                const updatedMetadata =
                    _omitMetadataKeys === '*'
                        ? { ...parsedMetadata } // Use only parsedMetadata if omitting all keys.
                        : omit(combinedMetadata, omitMetadataKeys) // Omit specific keys from the combined metadata.

                return {
                    ...doc,
                    metadata: updatedMetadata
                }
            })
        } else {
            // No additional metadata provided; only omit specified keys from existing metadata.
            docs = docs.map((doc) => {
                const updatedMetadata =
                    _omitMetadataKeys === '*'
                        ? {} // If omitting all keys, set metadata to an empty object.
                        : omit(doc.metadata, omitMetadataKeys) // Omit specific keys from the existing metadata.

                return {
                    ...doc,
                    metadata: updatedMetadata
                }
            })
        }

        if (isDebug) {
            options.logger.info(`Scrape completed`)
            options.logger.info(`Generated ${docs.length} total documents`)
            options.logger.info(`Processed ${processedURLs.size} unique URLs`)
            options.logger.info(`Encountered ${errorURLs.size} error responses...`)
            for (const item of errorURLs) options.logger.info(`URL: ${item[0]}, status: ${item[1]}`)
        }

        return docs
    }
}

module.exports = { nodeClass: Cheerio_DocumentLoaders }
