import { omit } from 'lodash'
import { ICommonObject, IDocument, INode, INodeData, INodeParams } from '../../../src/Interface'
import { TextSplitter } from 'langchain/text_splitter'
import { Browser, Page, PuppeteerWebBaseLoader, PuppeteerWebBaseLoaderOptions } from '@langchain/community/document_loaders/web/puppeteer'
import { test } from 'linkifyjs'
import { webCrawl, xmlScrape } from '../../../src'
import { PuppeteerLifeCycleEvent } from 'puppeteer'

class Puppeteer_DocumentLoaders implements INode {
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
        this.label = 'Puppeteer Web Scraper'
        this.name = 'puppeteerWebScraper'
        this.version = 1.0
        this.type = 'Document'
        this.icon = 'puppeteer.svg'
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
                label: 'Wait Until',
                name: 'waitUntilGoToOption',
                type: 'options',
                description: 'Select a go to wait until option',
                options: [
                    {
                        label: 'Load',
                        name: 'load',
                        description: `When the initial HTML document's DOM has been loaded and parsed`
                    },
                    {
                        label: 'DOM Content Loaded',
                        name: 'domcontentloaded',
                        description: `When the complete HTML document's DOM has been loaded and parsed`
                    },
                    {
                        label: 'Network Idle 0',
                        name: 'networkidle0',
                        description: 'Navigation is finished when there are no more than 0 network connections for at least 500 ms'
                    },
                    {
                        label: 'Network Idle 2',
                        name: 'networkidle2',
                        description: 'Navigation is finished when there are no more than 2 network connections for at least 500 ms'
                    }
                ],
                optional: true,
                additionalParams: true
            },
            {
                label: 'Wait for selector to load',
                name: 'waitForSelector',
                type: 'string',
                optional: true,
                additionalParams: true,
                description: 'CSS selectors like .div or #div'
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
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const isDebug = process.env.DEBUG === 'true'
        const textSplitter = nodeData.inputs?.textSplitter as TextSplitter
        const metadata = nodeData.inputs?.metadata
        const relativeLinksMethod = nodeData.inputs?.relativeLinksMethod as string
        const selectedLinks = nodeData.inputs?.selectedLinks as string[]

        let waitUntilGoToOption = nodeData.inputs?.waitUntilGoToOption as PuppeteerLifeCycleEvent
        let waitForSelector = nodeData.inputs?.waitForSelector as string
        const _omitMetadataKeys = nodeData.inputs?.omitMetadataKeys as string

        // NOTE: limit is always overriden to 3 when using the preview feature in the document store
        let limit = parseInt(`${nodeData.inputs?.limit}`, 10)
        if (isNaN(limit) || limit < 0) limit = 10

        let omitMetadataKeys: string[] = []
        if (_omitMetadataKeys) {
            omitMetadataKeys = _omitMetadataKeys.split(',').map((key) => key.trim())
        }

        let url = nodeData.inputs?.url as string
        url = url.trim()
        if (!test(url)) {
            throw new Error('Invalid URL')
        }

        async function puppeteerLoader(url: string, pageIndex: number, lastModified: number | null = null): Promise<any> {
            try {
                let docs = []
                const config: PuppeteerWebBaseLoaderOptions = {
                    launchOptions: {
                        args: ['--no-sandbox'],
                        headless: 'new'
                    }
                }
                if (waitUntilGoToOption) {
                    config['gotoOptions'] = {
                        waitUntil: waitUntilGoToOption
                    }
                }
                if (waitForSelector) {
                    config['evaluate'] = async (page: Page, _: Browser): Promise<string> => {
                        await page.waitForSelector(waitForSelector)

                        const result = await page.evaluate(() => document.body.innerHTML)
                        return result
                    }
                }
                const loader = new PuppeteerWebBaseLoader(url, config)
                if (textSplitter) {
                    docs = await loader.load()
                    docs = await textSplitter.splitDocuments(docs)
                } else {
                    docs = await loader.load()
                }
                return docs
            } catch (err) {
                if (isDebug) options.logger.error(`error in PuppeteerWebBaseLoader: ${err.message}, on page: ${url}`)
            }
        }

        let docs: IDocument[] = []

        if (relativeLinksMethod) {
            if (isDebug) options.logger.info(`Start ${relativeLinksMethod}`)

            // Determine pages to process based on selectedLinks and relativeLinksMethod.
            let pages: string[]
            let pagesLastMod: { [key: string]: number | null } = {}

            if (selectedLinks && selectedLinks.length > 0) {
                // If specific links are selected, use them up to the specified limit.
                pages = selectedLinks.slice(0, limit === 0 ? undefined : limit)
            } else if (relativeLinksMethod === 'webCrawl') {
                // If 'webCrawl' method is selected, fetch pages using web crawling.
                pages = await webCrawl(url, limit, true)
            } else {
                // Otherwise, fetch pages using XML scraping.
                pagesLastMod = (await xmlScrape(url, limit, true)) as { [key: string]: number | null }
                pages = Object.keys(pagesLastMod)
            }

            if (isDebug) options.logger.info(`pages: ${JSON.stringify(pages)}, length: ${pages.length}`)

            // If no pages were found, throw an error.
            if (!pages || pages.length === 0) {
                throw new Error('No relative links found')
            }

            // Process each page to extract content using cheerioLoader.
            for (const [index, page] of pages.entries()) {
                const loadedDocs = await puppeteerLoader(page, index, pagesLastMod[page])
                docs.push(...loadedDocs)
            }

            if (isDebug) options.logger.info(`Finish ${relativeLinksMethod}`)
        } else if (selectedLinks && selectedLinks.length > 0) {
            // If no relativeLinksMethod but selectedLinks are provided, process them.
            if (isDebug) options.logger.info(`pages: ${JSON.stringify(selectedLinks)}, length: ${selectedLinks.length}`)

            // Process each selected link up to the specified limit.
            for (const [index, page] of selectedLinks.slice(0, limit).entries()) {
                const loadedDocs = await puppeteerLoader(page, index)
                docs.push(...loadedDocs)
            }
        } else {
            // If neither relativeLinksMethod nor selectedLinks are provided, load from the base URL.
            docs = await puppeteerLoader(url, 0)
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

        return docs
    }
}

module.exports = { nodeClass: Puppeteer_DocumentLoaders }
