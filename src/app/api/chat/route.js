import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `You are an AI assistant for a RateMyProfessor-like platform. Your role is to help students find suitable professors based on their queries using a RAG (Retrieval-Augmented Generation) system. For each user question, you will provide information on the top 3 most relevant professors.

Your responses should follow this format:
1. Briefly restate the user's query to confirm understanding.
2. Present the top 3 professors in order of relevance, with each entry containing:
   - Professor's name
   - Subject they teach
   - Star rating (out of 5)
   - A short summary of their reviews
   - Why this professor might be a good fit based on the user's query

3. After presenting the top 3 options, offer a concise recommendation or additional advice related to the user's specific needs.

Remember to:
- Use the RAG system to retrieve and rank the most relevant professors based on the user's query.
- Provide balanced information, including both positive and negative aspects from reviews.
- Be objective and avoid personal biases.
- If the query is too vague or broad, ask for clarification to provide more accurate results.
- If there aren't enough relevant professors to match the query, explain this and provide the best alternatives available.

Your goal is to help students make informed decisions about their course selections based on professor reviews and ratings. Always maintain a helpful, informative, and impartial tone.`
require('dotenv').config();
const api_key = process.env.OPENAI_API_KEY
export async function POST(req){
    const data = await req.json()
        const pc = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
        })
        const index = pc.index('rag').namespace('ns1')
        const openai = new OpenAI()

        const text = data[data.length-1].content
        const embedding = await openai.embeddings.create({
            model: 'text-embedding-3-small',
            input: text,
            encoding_format: 'float',
        })
        const results = await index.query({
            topK: 3,
            includeMetadata: true,
            vector: embedding.data[0].embedding,
        })

        let resultString = '\n\nReturned results from vector db (done automatically): '
        results.matches.forEach((match)=>{
            resultString += `\n
            Professor: ${match.id}
            Review: ${match.metadata.stars}
            Subject: ${match.metadata.subject}
            Stars: ${match.metadata.stars}
            \n\n
            `
        })
        const lastMessage = data[data.length-1]
        const lastMessageContent = lastMessage.content + resultString
        const lastDataWithoutLastMessage = data.slice(0, data.length-1)
        const completion = await openai.chat.completions.create({
            messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent},
            ],
            model: 'gpt-4o',
            stream: true,
        })
        const stream = new ReadableStream({
                async start(controller){
                    const encoder = new TextEncoder()
                    try{
                        for await (const chuck of completion){
                            const content = chuck.choices[0]?.delta?.content
                            if(content){
                                const text = encoder.encode(content)
                                controller.enqueue(text)
                            }
                        }
                    }
                    catch(err){
                        controller.error(err)
                    } finally{
                        controller.close()
                    }
                },
        })

        return new NextResponse(stream)
}
