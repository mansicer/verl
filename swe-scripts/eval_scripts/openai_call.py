import re
import json
import io
import concurrent.futures
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import datasets
from datasets import Dataset, load_dataset
from tqdm import tqdm
from openai import OpenAI


def multithread_openai_call(
    client: OpenAI,
    messages: List[List[Dict[str, str]]],
    model_name: str,
    max_workers: int = 20,
    **kwargs
) -> List[str]:
    """Make parallel OpenAI API calls using multiple threads.
    
    Args:
        client: OpenAI client instance
        messages: List of message lists to process
        model_name: Name of the model to use
        max_workers: Maximum number of worker threads
        **kwargs: Additional arguments to pass to the API call
        
    Returns:
        List of responses in the same order as input messages
    """
    def call_openai(message: List[Dict[str, str]], **kwargs) -> str | List[str]:
        response = client.chat.completions.create(
            model=model_name, 
            messages=message,
            **kwargs
        )
        if hasattr(response, 'choices') and len(response.choices) > 1:
            return [choice.message.content for choice in response.choices]
        return response.choices[0].message.content

    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_message = {
            executor.submit(call_openai, message, **kwargs): i 
            for i, message in enumerate(messages)
        }
        
        with tqdm(total=len(messages), desc="Processing messages") as pbar:
            for future in concurrent.futures.as_completed(future_to_message):
                message_idx = future_to_message[future]
                response = future.result()
                responses.append((message_idx, response))
                pbar.update(1)
    
    responses.sort(key=lambda x: x[0])
    return [r[1] for r in responses]
