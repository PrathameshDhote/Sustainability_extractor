from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import json
import os

class IndicatorExtraction(BaseModel):
    """Structured output model for indicator extraction"""
    value: Optional[float] = Field(description="Numeric value of the indicator")
    unit: str = Field(description="Unit of measurement")
    source_quote: str = Field(description="Exact text from document supporting this extraction")
    page_number: int = Field(description="Page number where data was found")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    extraction_notes: str = Field(description="Any caveats or clarifications about the extraction")
    data_quality: str = Field(description="Quality assessment: 'high', 'medium', or 'low'")

class LLMProcessor:
    """
    Handles LLM-based extraction with verification
    
    UPDATED: Supports multiple LLM backends:
    - OpenAI (gpt-4o, gpt-3.5-turbo)
    - Anthropic (claude-3-5-sonnet, etc.)
    - Ollama (llama3.1:8b, mistral, etc.) - LOCAL & FREE
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        
        # Detect model type and initialize appropriate LLM
        if "gpt" in model_name.lower():
            # OpenAI models
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=4000
            )
            self.llm_type = "openai"
            
        elif "claude" in model_name.lower():
            # Anthropic models
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=4000
            )
            self.llm_type = "anthropic"
            
        else:
            # Ollama local models (default)
            print(f"  Using Ollama local model: {model_name}")
            print(f"  Make sure Ollama is running: ollama serve")
            
            self.llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                base_url="http://localhost:11434",  # Default Ollama port
                num_predict=4000  # Max tokens for output
            )
            self.llm_type = "ollama"
    
    def extract_from_table(self, table_markdown: str, indicator_name: str, 
                          expected_unit: str, esrs_ref: str, 
                          page_number: int) -> Optional[IndicatorExtraction]:
        """
        Extract indicator value from table markdown using LLM
        """
        prompt_template = """You are an expert sustainability data analyst extracting ESRS indicators from bank reports.

Extract the following indicator from this table:

Indicator: {indicator_name}
ESRS Reference: {esrs_ref}
Expected Unit: {expected_unit}
Source Page: {page_number}

Table Data (in markdown format):
{table_markdown}

CRITICAL INSTRUCTIONS:
1. Extract ONLY the 2024 value (or most recent year if 2024 not available)
2. If multiple values exist (e.g., different scopes or boundaries), extract the CONSOLIDATED/TOTAL value
3. Return the EXACT quote from the table that contains this value
4. Assign confidence score:
   - 1.0 = Clear table cell with explicit label and value
   - 0.8 = Value found but label is ambiguous
   - 0.6 = Value inferred from context
   - 0.4 = Multiple possible values, best guess
   - 0.0 = Cannot find value

5. If the value cannot be found, return null for value and confidence_score of 0.0

Return ONLY valid JSON in this exact format:
{{
    "value": <numeric_value or null>,
    "unit": "<unit>",
    "source_quote": "<exact text from document>",
    "page_number": <page_number>,
    "confidence_score": <0.0 to 1.0>,
    "extraction_notes": "<any clarifications>",
    "data_quality": "high|medium|low"
}}

Do not include any explanation or markdown formatting, just the raw JSON."""

        formatted_prompt = prompt_template.format(
            indicator_name=indicator_name,
            esrs_ref=esrs_ref,
            expected_unit=expected_unit,
            page_number=page_number,
            table_markdown=table_markdown[:6000]  # Limit for local models
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            
            # Extract content based on LLM type
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Parse JSON response
            result = self._parse_json_response(content)
            
            if result:
                # Convert to IndicatorExtraction object
                extraction = IndicatorExtraction(**result)
                extraction = self._validate_extraction(extraction, expected_unit)
                return extraction
            
            return None
            
        except Exception as e:
            print(f"  ⚠ Error in table extraction: {e}")
            return None
    
    def extract_from_narrative(self, text_context: str, indicator_name: str,
                               expected_unit: str, esrs_ref: str,
                               page_number: int, 
                               search_keywords: List[str]) -> Optional[IndicatorExtraction]:
        """
        Extract indicator from narrative text using LLM
        Used for governance indicators and qualitative data
        """
        prompt_template = """You are an expert sustainability analyst extracting governance and qualitative indicators from bank reports.

Extract the following indicator from this narrative text:

Indicator: {indicator_name}
ESRS Reference: {esrs_ref}
Expected Unit: {expected_unit}
Keywords to look for: {keywords}
Source Page: {page_number}

Text Context:
{text_context}

CRITICAL INSTRUCTIONS:
1. Look for explicit statements about {indicator_name}
2. Extract the 2024 value or most recent disclosed value
3. Return the EXACT SENTENCE that contains this information as source_quote
4. For qualitative targets (e.g., "net zero by 2050"), extract the numeric part (2050)
5. Assign confidence based on clarity of disclosure:
   - 1.0 = Explicit numerical statement
   - 0.7 = Clearly stated but requires interpretation
   - 0.5 = Implied from context
   - 0.0 = Not found

Return ONLY valid JSON in this exact format:
{{
    "value": <numeric_value or null>,
    "unit": "<unit>",
    "source_quote": "<exact sentence from document>",
    "page_number": <page_number>,
    "confidence_score": <0.0 to 1.0>,
    "extraction_notes": "<any clarifications>",
    "data_quality": "high|medium|low"
}}

Do not include any explanation, just the raw JSON."""

        formatted_prompt = prompt_template.format(
            indicator_name=indicator_name,
            esrs_ref=esrs_ref,
            expected_unit=expected_unit,
            keywords=", ".join(search_keywords),
            page_number=page_number,
            text_context=text_context[:8000]  # Larger context for narrative
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            result = self._parse_json_response(content)
            
            if result:
                extraction = IndicatorExtraction(**result)
                extraction = self._validate_extraction(extraction, expected_unit)
                return extraction
            
            return None
            
        except Exception as e:
            print(f"  ⚠ Error in narrative extraction: {e}")
            return None
    
    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """
        Parse JSON from LLM response, handling various formats
        """
        # Try direct JSON parse first
        try:
            return json.loads(content)
        except:
            pass
        
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except:
                pass
        
        if "```" in content:
            try:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except:
                pass
        
        # Try to find JSON object in text
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
        except:
            pass
        
        print(f"  ⚠ Could not parse JSON from response: {content[:200]}")
        return None
    
    def verify_extraction(self, extraction: IndicatorExtraction, 
                         original_text: str) -> float:
        """
        Verify an extraction by checking if source_quote exists in original text
        Returns adjusted confidence score
        """
        if not extraction.source_quote:
            return extraction.confidence_score * 0.5
        
        quote_lower = extraction.source_quote.lower()
        text_lower = original_text.lower()
        
        if quote_lower in text_lower:
            return extraction.confidence_score
        
        quote_words = set(quote_lower.split())
        text_words = set(text_lower.split())
        
        overlap = len(quote_words.intersection(text_words)) / len(quote_words) if quote_words else 0
        
        if overlap > 0.5:
            return extraction.confidence_score * 0.9
        else:
            return extraction.confidence_score * 0.6
    
    def _validate_extraction(self, extraction: IndicatorExtraction, 
                            expected_unit: str) -> IndicatorExtraction:
        """Validate and clean extraction result"""
        
        # Check unit consistency (flexible for local models)
        if extraction.unit.lower() != expected_unit.lower():
            # Check if units are similar (e.g., "tCO2e" vs "tco2e")
            if extraction.unit.replace(" ", "").lower() != expected_unit.replace(" ", "").lower():
                extraction.extraction_notes += f" | Unit mismatch: expected {expected_unit}, got {extraction.unit}"
                extraction.confidence_score *= 0.8
        
        # Validate confidence range
        extraction.confidence_score = max(0.0, min(1.0, extraction.confidence_score))
        
        # Assign data quality
        if extraction.confidence_score >= 0.8:
            extraction.data_quality = "high"
        elif extraction.confidence_score >= 0.5:
            extraction.data_quality = "medium"
        else:
            extraction.data_quality = "low"
        
        return extraction
    
    def batch_extract(self, contexts: List[Dict[str, Any]], 
                     indicator_name: str, expected_unit: str,
                     esrs_ref: str) -> List[IndicatorExtraction]:
        """
        Batch extraction for multiple contexts
        Returns list of extractions sorted by confidence
        """
        results = []
        
        for ctx in contexts:
            if ctx.get("type") == "table":
                result = self.extract_from_table(
                    ctx["content"], indicator_name, expected_unit,
                    esrs_ref, ctx["page_number"]
                )
            else:  # narrative
                result = self.extract_from_narrative(
                    ctx["content"], indicator_name, expected_unit,
                    esrs_ref, ctx["page_number"], ctx.get("keywords", [])
                )
            
            if result and result.value is not None:
                results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return results