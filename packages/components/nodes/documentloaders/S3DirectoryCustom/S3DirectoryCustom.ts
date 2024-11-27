const { nodeClass: S3Directory } = require('../S3Directory/S3Directory')
import { ICommonObject, IDocument, INodeData } from '../../../src/Interface'
import { LoadersMapping } from 'langchain/document_loaders/fs/directory'
import { TextLoader } from 'langchain/document_loaders/fs/text'

class S3DirectoryCustom extends S3Directory {
    constructor() {
        super()

        this.label = 'S3 Directory Custom'
        this.name = 's3DirectoryCustom'
        this.version = 0.1
        this.description = 'Load Data from S3 Buckets with Custom Functionality'
        this.baseClasses = [this.type]
    }

    /**
     * Override the processDocuments method to add custom processing.
     *
     * @param docs - The array of documents to process.
     * @returns The processed array of documents.
     */
    protected processDocuments(docs: IDocument[], nodeData: INodeData, options: ICommonObject): IDocument[] {
        const debug = process.env.DEBUG === 'true'

        if (debug) {
            options.logger.info(`Processing ${docs.length} documents...`)
            options.logger.info(`[0] metadata ${docs[0].metadata}`)
            options.logger.info(`[0] content ${docs[0].pageContent}`)
        }

        const docsByKey: Record<string, IDocument> = {}
        const metadataByKey: Record<string, any> = {}

        // docs is an array of IDocument objects that contains both metadata and content docuemnts
        // we want to combine the metadata and content for each document into a single document object
        // first we will create a map of source keys to metadata objects, then we will iterate over the documents and add the metadata to the document
        // and add these new documents to a new array that we will return

        // create a map of source keys to metadata objects
        docs.forEach((doc) => {
            const key = doc.metadata.source as string
            if (key.endsWith('.metadata.json')) {
                // it's a metadata file
                const contentKey = key.substring(0, key.length - 14)
                // store the metadata mapped to the content key
                const parsed = JSON.parse(doc.pageContent)
                if (parsed['metadataAttributes']) metadataByKey[contentKey] = parsed['metadataAttributes']
                else metadataByKey[contentKey] = parsed
            } else {
                // it's a content file, store the document mapped to the content key
                docsByKey[key] = doc
            }
        })

        // iterate over the documents and add the metadata to the document
        const newDocs: IDocument[] = []
        Object.keys(docsByKey).forEach((key) => {
            const doc = docsByKey[key]
            const metadata = metadataByKey[key]
            if (metadata) {
                doc.metadata = { ...doc.metadata, ...metadata }
            } else {
                if (debug) {
                    options.logger.warning(`No metadata found for ${key}`)
                }
            }
            newDocs.push(doc)
        })

        return newDocs
    }

    /**
     * Override the getLoaders method to customize document loaders.
     *
     * @param pdfUsage - Determines how PDFs are handled ('perFile' or 'perPage').
     * @returns A mapping of file extensions to their respective document loaders.
     */
    protected getLoaders(pdfUsage: string = 'perFile'): LoadersMapping {
        const loaders = super.getLoaders(pdfUsage)

        // replace the '.json' loader with a basic text loader
        // - we're going to parse the JSON ourselves later rather than letting the loader do it
        loaders['.json'] = (path: string) => new TextLoader(path)

        return loaders
    }
}

module.exports = { nodeClass: S3DirectoryCustom }
