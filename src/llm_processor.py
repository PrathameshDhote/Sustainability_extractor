from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
import json
import os
import re


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
    
    FIXED: Robust JSON parsing with aggressive cleanup for:
    - Commas in numbers (1,074,786 → 1074786)
    - Percent signs (43% → 43.0)
    - String 'null' → JSON null
    - Row confusion prevention
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.extracted_values = {}  # Track values to detect duplicates
        
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
                base_url="http://localhost:11434",
                num_predict=6000,  # Increased to prevent cutoffs
                format='json'      # Force JSON output format
            )
            self.llm_type = "ollama"
    
    def extract_from_table(self, table_markdown: str, indicator_name: str, 
                          expected_unit: str, esrs_ref: str, 
                          page_number: int) -> Optional[IndicatorExtraction]:
        """
        Extract indicator value from table markdown using LLM with precision auditing
        """
        prompt_template = """You are a precision sustainability auditor extracting ESRS indicators from banking reports.

EXTRACTION TASK:
Indicator: {indicator_name}
ESRS Reference: {esrs_ref}
Expected Unit: {expected_unit}
Source Page: {page_number}

TABLE CONTEXT (Page {page_number}):
{table_markdown}

CRITICAL AUDIT RULES:
1. Locate the EXACT row that matches "{indicator_name}" - not similar rows, not total rows
2. Find the column for year 2024 (or most recent year if 2024 not available)
3. If the table lists multiple years (2022, 2023, 2024), ensure you DO NOT pull 2023 or 2022 data
4. Verify the row label BEFORE extracting:
   - If looking for "Scope 2" emissions, DO NOT extract from "Scope 1" or "Scope 3" rows
   - If looking for "Total Energy", DO NOT extract from "Scope 1 emissions" rows
   - If looking for "Female Employees %", DO NOT extract from "Gender Pay Gap" rows
5. If the value is "1,074,786" but the row says "Total Assets" or "Scope 1", it is WRONG for other indicators

COMMON MISTAKES TO AVOID:
❌ Extracting the same value for different indicators
❌ Mixing up Scope 1, 2, and 3 emissions
❌ Using 2023 data when 2024 is available
❌ Extracting from "Total" rows when specific indicator is available
❌ Confusing similar indicators (e.g., "Gender Pay Gap" vs "Female Employees")

JSON FORMAT REQUIREMENTS (STRICT):
- "value": Must be a NUMBER ONLY (e.g., 1074786.0). NO commas. NO percent signs. NO currency symbols.
  Examples: 
    CORRECT: 1074786.0
    CORRECT: 43.0 (for 43%)
    WRONG: "1,074,786"
    WRONG: "43%"
    WRONG: "null" (use null without quotes)
- "unit": Must match "{expected_unit}" exactly
- "source_quote": The EXACT text of the row you selected, including the value
- "page_number": {page_number}
- "confidence_score": 
    1.0 if row label matches perfectly and value is from 2024
    0.8 if row label is close match
    0.5 if you had to guess between similar rows
    0.0 if you cannot find the indicator
- "extraction_notes": Explain which row and column you used (e.g., "Row: Scope 2 emissions, Column: 2024")
- "data_quality": "high" (not "hi"), "medium" (not "med"), or "low"

If you cannot find the EXACT indicator "{indicator_name}" in the table:
- Set "value": null (not "null" as a string, but actual JSON null)
- Set "confidence_score": 0.0

Return ONLY raw JSON (no markdown, no explanation):
{{
    "value": <numeric_value_without_commas or null>,
    "unit": "{expected_unit}",
    "source_quote": "<exact row text>",
    "page_number": {page_number},
    "confidence_score": <0.0 to 1.0>,
    "extraction_notes": "<which row and column>",
    "data_quality": "high"
}}"""

        formatted_prompt = prompt_template.format(
            indicator_name=indicator_name,
            esrs_ref=esrs_ref,
            expected_unit=expected_unit,
            page_number=page_number,
            table_markdown=table_markdown[:6000]
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            
            # Extract content based on LLM type
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Parse JSON response with aggressive cleanup
            result = self._parse_json_response(content)
            
            if result:
                # Convert to IndicatorExtraction object
                extraction = IndicatorExtraction(**result)
                extraction = self._validate_extraction(extraction, expected_unit, indicator_name)
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
        """
        prompt_template = """You are a precision sustainability auditor extracting governance indicators from narrative text.

EXTRACTION TASK:
Indicator: {indicator_name}
ESRS Reference: {esrs_ref}
Expected Unit: {expected_unit}
Keywords: {keywords}
Source Page: {page_number}

TEXT CONTEXT:
{text_context}

CRITICAL INSTRUCTIONS:
1. Look for explicit statements about "{indicator_name}"
2. Extract the 2024 value or most recent disclosed value
3. Return the EXACT SENTENCE that contains this information
4. For qualitative targets (e.g., "net zero by 2050"), extract the year (2050)
5. For percentages (e.g., "43% female board members"), extract the number only (43.0)

JSON FORMAT REQUIREMENTS (STRICT):
- "value": Must be a NUMBER ONLY. NO commas. NO percent signs.
  Examples:
    CORRECT: 2050.0 (for "net zero by 2050")
    CORRECT: 43.0 (for "43%")
    WRONG: "2,050"
    WRONG: "43%"
- "unit": "{expected_unit}"
- "source_quote": The exact sentence containing the value
- "confidence_score":
    1.0 = Explicit numerical statement
    0.7 = Clearly stated but requires interpretation
    0.5 = Implied from context
    0.0 = Not found (set "value": null)

Return ONLY raw JSON:
{{
    "value": <numeric_value or null>,
    "unit": "{expected_unit}",
    "source_quote": "<exact sentence>",
    "page_number": {page_number},
    "confidence_score": <0.0 to 1.0>,
    "extraction_notes": "<clarification>",
    "data_quality": "high"
}}"""

        formatted_prompt = prompt_template.format(
            indicator_name=indicator_name,
            esrs_ref=esrs_ref,
            expected_unit=expected_unit,
            keywords=", ".join(search_keywords),
            page_number=page_number,
            text_context=text_context[:8000]
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
                extraction = self._validate_extraction(extraction, expected_unit, indicator_name)
                return extraction
            
            return None
            
        except Exception as e:
            print(f"  ⚠ Error in narrative extraction: {e}")
            return None
    
    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """
        Aggressive cleanup for local LLM output.
        Ensures 'value' is a clean numeric float before Pydantic validation.
        
        Handles:
        - Commas in numbers: "1,074,786" → 1074786.0
        - Percent signs: "43%" → 43.0
        - String 'null' → JSON null
        - Truncated strings: "hi" → "high"
        """
        content_cleaned = content.strip()
        
        # 1. Standard Markdown/Codeblock Cleanup
        if "```json" in content_cleaned:
            content_cleaned = content_cleaned.split("```json").split("``` ")
        elif "```" in content_cleaned:
            content_cleaned = content_cleaned.split("``````")[0].strip()

        # 2. Extract the JSON object if it's buried in text
        start = content_cleaned.find("{")
        end = content_cleaned.rfind("}") + 1
        if start != -1 and end > start:
            content_cleaned = content_cleaned[start:end]

        try:
            data = json.loads(content_cleaned)
            
            # 3. CRITICAL: Clean the 'value' field specifically
            val = data.get('value')
            if val is not None:
                if isinstance(val, str):
                    # Remove commas, spaces, percent signs, and currency symbols
                    clean_val = re.sub(r'[,\s%$€£]', '', val)
                    
                    # Handle string 'null' or empty string
                    if clean_val.lower() == 'null' or clean_val == '':
                        data['value'] = None
                    else:
                        try:
                            # Convert to float
                            data['value'] = float(clean_val)
                        except ValueError:
                            # If conversion fails, set to None
                            data['value'] = None
                            print(f"  ⚠ Could not convert value to float: '{val}'")
            
            # 4. Handle string 'null' for the whole value field
            if data.get('value') == 'null':
                data['value'] = None
            
            # 5. Fix truncated data_quality strings
            if 'data_quality' in data:
                quality_map = {
                    'hi': 'high', 'h': 'high',
                    'med': 'medium', 'me': 'medium', 'm': 'medium',
                    'lo': 'low', 'l': 'low'
                }
                dq = str(data['data_quality']).lower()
                if dq in quality_map:
                    data['data_quality'] = quality_map[dq]
            
            # 6. Ensure required fields exist
            required_fields = ['value', 'unit', 'source_quote', 'page_number', 
                             'confidence_score', 'extraction_notes', 'data_quality']
            for field in required_fields:
                if field not in data:
                    if field == 'value':
                        data[field] = None
                    elif field in ['unit', 'source_quote', 'extraction_notes', 'data_quality']:
                        data[field] = ''
                    elif field == 'page_number':
                        data[field] = 0
                    elif field == 'confidence_score':
                        data[field] = 0.0

            return data
            
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON Decode Error: {e}")
            
            # Last resort: Try to manually extract values using regex
            try:
                value_match = re.search(r'"value":\s*(["\']?)(\d{1,3}(?:,\d{3})*\.?\d*)%?\1', content_cleaned)
                unit_match = re.search(r'"unit":\s*"([^"]+)"', content_cleaned)
                page_match = re.search(r'"page_number":\s*(\d+)', content_cleaned)
                quote_match = re.search(r'"source_quote":\s*"([^"]*)"', content_cleaned)
                conf_match = re.search(r'"confidence_score":\s*([\d.]+)', content_cleaned)
                
                if value_match and unit_match:
                    value_str = value_match.group(2).replace(',', '')
                    
                    return {
                        'value': float(value_str) if value_str else None,
                        'unit': unit_match.group(1),
                        'source_quote': quote_match.group(1) if quote_match else 'Reconstructed from partial response',
                        'page_number': int(page_match.group(1)) if page_match else 0,
                        'confidence_score': float(conf_match.group(1)) if conf_match else 0.5,
                        'extraction_notes': 'Reconstructed from malformed JSON',
                        'data_quality': 'medium'
                    }
            except Exception as reconstruct_error:
                print(f"  ⚠ Reconstruction failed: {reconstruct_error}")
            
            print(f"  ⚠ Could not parse JSON from response: {content[:300]}")
            return None
        
        except Exception as e:
            print(f"  ⚠ Unexpected error in JSON parsing: {e}")
            return None
    
    def _check_duplicate_value(self, indicator_name: str, value: float) -> float:
        """
        Check if this exact value was extracted for a different indicator.
        Returns adjusted confidence score to penalize likely copy errors.
        """
        if value is None:
            return 1.0
        
        # Create a key for this value (rounded to avoid floating point issues)
        value_key = round(value, 2)
        
        if value_key in self.extracted_values:
            previous_indicator = self.extracted_values[value_key]
            if previous_indicator != indicator_name:
                # Same value for different indicator - likely copy error
                print(f"  ⚠ Duplicate value detected: {value} used for both '{previous_indicator}' and '{indicator_name}'")
                return 0.3  # Severely reduce confidence
        
        self.extracted_values[value_key] = indicator_name
        return 1.0  # No duplicate detected
    
    def verify_extraction(self, extraction: IndicatorExtraction, 
                         original_text: str) -> float:
        """
        Verify an extraction by checking if source_quote exists in original text
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
                            expected_unit: str, indicator_name: str = "") -> IndicatorExtraction:
        """
        Validate and clean extraction result with duplicate detection
        """
        # Check for duplicate values across different indicators
        if extraction.value is not None and indicator_name:
            dup_confidence = self._check_duplicate_value(indicator_name, extraction.value)
            extraction.confidence_score *= dup_confidence
            if dup_confidence < 1.0:
                extraction.extraction_notes += " | DUPLICATE VALUE DETECTED - likely extraction error"
                extraction.data_quality = "low"
        
        # Check unit consistency
        if extraction.unit.lower() != expected_unit.lower():
            # Check if units are similar (e.g., "tCO2e" vs "tco2e")
            if extraction.unit.replace(" ", "").lower() != expected_unit.replace(" ", "").lower():
                extraction.extraction_notes += f" | Unit mismatch: expected {expected_unit}, got {extraction.unit}"
                extraction.confidence_score *= 0.8
        
        # Validate confidence range
        extraction.confidence_score = max(0.0, min(1.0, extraction.confidence_score))
        
        # Assign data quality based on confidence (unless already marked as low due to duplicates)
        if extraction.data_quality != "low":
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
