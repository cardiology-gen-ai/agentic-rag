#!/usr/bin/env python3
"""
Script to view conversation data from the database.
"""

import json
import psycopg2
from datetime import datetime

def connect_db():
    """Connect to the database"""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="root",
        password="root"
    )

def view_recent_conversations(limit=5):
    """View recent conversations from agent_states table"""
    conn = connect_db()
    cur = conn.cursor()
    
    # Get recent agent states (where the actual conversation data is stored)
    cur.execute("""
        SELECT thread_id, state_data, created_at
        FROM agent_states
        WHERE state_type = 'general'
        ORDER BY created_at DESC
        LIMIT %s;
    """, (limit,))
    
    results = cur.fetchall()
    
    if not results:
        print("No conversations found in database.")
        return
    
    for i, (thread_id, state_data, created_at) in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Conversation {i}")
        print(f"Thread ID: {thread_id}")
        print(f"Created: {created_at}")
        print(f"{'='*60}")
        
        if state_data and 'messages' in state_data:
            messages = state_data['messages']
            print(f"Messages ({len(messages)}):")
            
            for j, msg in enumerate(messages, 1):
                msg_type = msg.get('type', 'Unknown')
                content = msg.get('content', '')
                
                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                
                print(f"  {j}. {msg_type}: {content}")
        
        if state_data and 'conversation_summary' in state_data:
            summary = state_data.get('conversation_summary')
            if summary:
                print(f"\nSummary: {summary}")
    
    cur.close()
    conn.close()

def view_conversation_by_thread_id(thread_id):
    """View a specific conversation by thread ID"""
    conn = connect_db()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT state_data, created_at
        FROM agent_states
        WHERE thread_id = %s AND state_type = 'general'
        ORDER BY created_at DESC
        LIMIT 1;
    """, (thread_id,))
    
    result = cur.fetchone()
    
    if not result:
        print(f"No conversation found for thread ID: {thread_id}")
        return
    
    state_data, created_at = result
    
    print(f"Thread ID: {thread_id}")
    print(f"Created: {created_at}")
    print("="*60)
    
    if state_data and 'messages' in state_data:
        messages = state_data['messages']
        print(f"Full conversation ({len(messages)} messages):\n")
        
        for i, msg in enumerate(messages, 1):
            msg_type = msg.get('type', 'Unknown')
            content = msg.get('content', '')
            
            print(f"{i}. [{msg_type}] {content}\n")
    
    if state_data and 'conversation_summary' in state_data:
        summary = state_data.get('conversation_summary')
        if summary:
            print(f"Summary: {summary}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # View specific conversation by thread ID
        thread_id = sys.argv[1]
        view_conversation_by_thread_id(thread_id)
    else:
        # View recent conversations
        view_recent_conversations()