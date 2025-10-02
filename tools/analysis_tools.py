import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
import re

# LangChain import for creating the tool
from langchain_core.tools import tool

# Your existing imports from the original files
# Ensure these helper modules (gemma_llm, groq_llm, plotting_utils) are accessible in your project
try:
    from gemma_llm import GemmaLLM
    from groq_llm import GroqLLM
    from utils.plotting_utils import plot_histogram, plot_bar, plot_scatter
except ImportError:
    print("Warning: Could not import LLM helpers or plotting utils. Functionality will be limited.")
    # Define dummy classes if imports fail to allow the script to load
    class BaseLLM:
        def __init__(self, **kwargs): pass
        def is_available(self): return False
        def __call__(self, prompt): return "LLM not available."
    GemmaLLM = GroqLLM = BaseLLM
    def plot_histogram(df, col): return "Plotting disabled."
    def plot_bar(df, col1, col2): return "Plotting disabled."
    def plot_scatter(df, col1, col2): return "Plotting disabled."


# ================== THE LANGCHAIN TOOL ==================
# This is the single, stateless function that your "Insight Analyst" agent will call.

@tool
def analyze_user_question(question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Answers a user's question about the dataset by first classifying the question
    (statistical, analytical, descriptive) and then routing it to the best analysis engine.
    It generates a textual answer and creates a visualization if deemed necessary.
    This is the primary tool for all data analysis queries.
    
    Args:
        question (str): The user's question about the data.
        df (pd.DataFrame): The pandas DataFrame containing the data.
        dataset_name (str): The name of the dataset being analyzed.
        
    Returns:
        dict: A dictionary containing the analysis results, including 'answer', 
              'visualization_html', 'visualization_json', and classification details.
    """
    print(f"🔬 Tool 'analyze_user_question' starting for question: '{question}'")
    try:
        engine = UnifiedAnalysisEngine()
        result = engine.analyze_question(question, df, dataset_name)

        print("✅ Tool 'analyze_user_question' finished successfully.")
        return result
        
    except Exception as e:
        print(f"❌ Error in 'analyze_user_question' tool: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "answer": "An error occurred during analysis."}


# ========== INTERNAL CLASSES AND LOGIC ==========
# All the complex logic from your original files is now encapsulated here.
# These classes are not tools themselves but are used by the main tool above.

class QuestionType(Enum):
    STATISTICAL = "statistical"
    ANALYTICAL = "analytical"
    DESCRIPTIVE = "descriptive"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"

@dataclass
class ClassificationResult:
    question_type: QuestionType
    confidence: float
    requires_visualization: bool
    visualization_type: str
    reasoning: str

class QuestionClassifier:
    """Clean, simple question classifier from your original file."""
    
    def classify(self, question: str, df: pd.DataFrame) -> ClassificationResult:
        q_lower = question.lower()
        
        statistical_keywords = ['calculate', 'compute', 'sum', 'total', 'average', 'mean', 'median', 'min', 'max', 'count', 'how many', 'top', 'bottom', 'rank', 'percentage']
        analytical_keywords = ['analyze', 'trend', 'pattern', 'relationship', 'correlation', 'impact', 'influence', 'why', 'how does']
        descriptive_keywords = ['what is', 'what are', 'who', 'describe', 'explain', 'show me', 'list', 'summary']
        
        stat_score = sum(1 for kw in statistical_keywords if kw in q_lower)
        analytical_score = sum(1 for kw in analytical_keywords if kw in q_lower)
        desc_score = sum(1 for kw in descriptive_keywords if kw in q_lower)
        
        scores = {
            QuestionType.STATISTICAL: stat_score,
            QuestionType.ANALYTICAL: analytical_score,
            QuestionType.DESCRIPTIVE: desc_score,
        }
        
        primary_type = max(scores, key=scores.get) if any(s > 0 for s in scores.values()) else QuestionType.DESCRIPTIVE
        total = sum(scores.values())
        confidence = (scores[primary_type] / total) if total > 0 else 0.5
        
        viz_needed = self._needs_visualization(q_lower, primary_type)
        viz_type = self._suggest_visualization_type(q_lower, primary_type, df) if viz_needed else "none"
        
        return ClassificationResult(
            question_type=primary_type,
            confidence=confidence,
            requires_visualization=viz_needed,
            visualization_type=viz_type,
            reasoning=f"Detected {primary_type.value} question with {confidence:.1%} confidence"
        )
    
    def _needs_visualization(self, question: str, q_type: QuestionType) -> bool:
        no_viz_keywords = ['how many columns', 'data types', 'shape', 'info']
        if any(kw in question for kw in no_viz_keywords):
            return False
        if q_type == QuestionType.STATISTICAL:
            return True
        if q_type == QuestionType.ANALYTICAL:
            return any(kw in question for kw in ['trend', 'pattern', 'distribution', 'relationship'])
        if q_type == QuestionType.DESCRIPTIVE:
            return 'distribution' in question or 'spread' in question
        return False
    
    def _suggest_visualization_type(self, question: str, q_type: QuestionType, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if any(word in question for word in ['over time', 'trend']):
            return 'line_chart'
        if 'distribution' in question:
            return 'histogram'
        if any(word in question for word in ['compare', 'versus', 'top', 'bottom', 'rank']):
            return 'bar_chart'
        if any(word in question for word in ['relationship', 'correlation']) and len(numeric_cols) >= 2:
            return 'scatter_plot'
        return 'bar_chart'

class HybridCalculationEngine:
    """
    A real implementation of the HybridCalculationEngine.
    It uses rule-based pandas for exact calculations based on keywords.
    """
    def analyze_with_calculations(self, question: str, df: pd.DataFrame) -> str:
        q_lower = question.lower()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            return f"No numeric columns available for calculation. Total rows: {len(df)}."

        target_col = numeric_cols[0] # Default to first numeric column

        # Try to find a specific column mentioned in the question
        for col in numeric_cols:
            if col.lower() in q_lower:
                target_col = col
                break
        
        if any(k in q_lower for k in ['average', 'mean']):
            result = df[target_col].mean()
            return f"The average for '{target_col}' is **{result:,.2f}**."
        
        if any(k in q_lower for k in ['total', 'sum']):
            result = df[target_col].sum()
            return f"The sum for '{target_col}' is **{result:,.2f}**."

        if any(k in q_lower for k in ['max', 'maximum', 'highest']):
            result = df[target_col].max()
            return f"The maximum value for '{target_col}' is **{result:,.2f}**."

        if any(k in q_lower for k in ['min', 'minimum', 'lowest']):
            result = df[target_col].min()
            return f"The minimum value for '{target_col}' is **{result:,.2f}**."
            
        if 'count' in q_lower:
            return f"The total count of records is **{len(df):,}**."
            
        if 'correlation' in q_lower and len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            corr = df[col1].corr(df[col2])
            return f"The correlation between '{col1}' and '{col2}' is **{corr:.3f}**."

        return f"Performed a basic statistical check. For example, the mean of '{target_col}' is {df[target_col].mean():,.2f}."

class InsightAgent:
    """The full implementation of your InsightAgent from visualizationinsightsnode.py."""
    def __init__(self, model: str = "google/gemma-2-9b-it"):
        self.llm = GemmaLLM(model_name=model, temperature=0.2, max_tokens=2000)
        self.last_figure_json = None

    def answer(self, df: pd.DataFrame, question: str, dataset_name: str) -> dict:
        if not self.llm.is_available():
            return {"answer": "LLM for insights is not available."}
            
        data_context = self._create_data_context(df)
        
        prompt = f"""You are a senior data analyst.
        
        QUESTION: {question}

        DATASET CONTEXT:
        {data_context}

        Provide a comprehensive, data-driven answer. Use specific column names and patterns from the data.
        Format your answer clearly with bullet points.
        """
        
        answer = self.llm(prompt)
        
        viz_html, viz_json = self._create_visualization(df, question)
        self.last_figure_json = viz_json

        return {
            "question": question,
            "answer": answer,
            "visualization_html": viz_html,
            "visualization_json": viz_json,
        }

    def _create_data_context(self, df: pd.DataFrame) -> str:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        context = f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        if numeric_cols:
            context += f"Numeric columns: {numeric_cols[:5]}\n"
        if categorical_cols:
            context += f"Categorical columns: {categorical_cols[:5]}\n"
        context += f"Sample Data:\n{df.head(3).to_string()}"
        return context

    def _create_visualization(self, df: pd.DataFrame, question: str):
        try:
            import plotly.express as px
            import plotly.io as pio

            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            q_lower = question.lower()
            fig = None
            
            if 'distribution' in q_lower and numeric_cols:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
            elif 'relationship' in q_lower and len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}")
            elif categorical_cols and numeric_cols:
                # Default to a bar chart for general questions
                grouped = df.groupby(categorical_cols[0])[numeric_cols[0]].mean().reset_index()
                fig = px.bar(grouped, x=categorical_cols[0], y=numeric_cols[0], title=f"Average {numeric_cols[0]} by {categorical_cols[0]}")
            elif numeric_cols:
                 fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
            
            if fig:
                html = fig.to_html(full_html=False, include_plotlyjs='cdn')
                json_fig = pio.to_json(fig)
                return html, json_fig
        except Exception as e:
            print(f"Visualization creation failed: {e}")
        
        return None, None

class UnifiedAnalysisEngine:
    """The full implementation of your UnifiedAnalysisEngine."""
    
    def __init__(self):
        self.classifier = QuestionClassifier()
        # The engine instantiates its required components internally
        self.hybrid_engine = HybridCalculationEngine()
        self.insight_agent = InsightAgent()
        
    def analyze_question(self, question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        classification = self.classifier.classify(question, df)
        print(f"🎯 Engine Classification: {classification.question_type.value.upper()}")
        
        try:
            if classification.question_type == QuestionType.STATISTICAL:
                result = self._handle_statistical_question(question, df, classification)
            elif classification.question_type == QuestionType.ANALYTICAL:
                result = self._handle_analytical_question(question, df, dataset_name, classification)
            else:  # DESCRIPTIVE
                result = self._handle_descriptive_question(question, df, dataset_name)
            
            result['classification'] = {
                'type': classification.question_type.value,
                'confidence': classification.confidence,
                'reasoning': classification.reasoning,
                'requires_visualization': classification.requires_visualization,
                'visualization_type': classification.visualization_type,
            }
            result['question_type'] = classification.question_type.value
            return result
        except Exception as e:
            print(f"❌ Analysis routing failed: {e}")
            return self._handle_fallback_analysis(question, df)
    
    def _handle_statistical_question(self, question: str, df: pd.DataFrame, classification: ClassificationResult) -> Dict[str, Any]:
        print("⚡ Using Hybrid Calculation Engine for statistical analysis...")
        calculation_result = self.hybrid_engine.analyze_with_calculations(question, df)
        
        ia_result = self.insight_agent.answer(df, question, "statistical_context")
        
        combined_answer = f"⚡ **Statistical Calculation**\n{calculation_result}\n\n---\n\n🧠 **Analytical Insight**\n{ia_result['answer']}"
        
        return {
            'question': question, 'answer': combined_answer,
            'visualization_html': ia_result.get('visualization_html'),
            'visualization_json': ia_result.get('visualization_json'),
            'method': 'statistical_hybrid_engine+insights'
        }
    
    def _handle_analytical_question(self, question: str, df: pd.DataFrame, dataset_name: str, classification: ClassificationResult) -> Dict[str, Any]:
        print("🧠 Using Insight Agent for analytical analysis...")
        result = self.insight_agent.answer(df, question, dataset_name)
        result['answer'] = f"🧠 **Analytical Insights**\n\n{result.get('answer', '')}"
        result['method'] = 'analytical_insight_agent'
        return result
    
    def _handle_descriptive_question(self, question: str, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        print("📋 Using Descriptive Summarizer...")
        q_lower = question.lower()
        if 'columns' in q_lower or 'variables' in q_lower:
            answer = f"Dataset contains {len(df.columns)} columns: {', '.join(df.columns.tolist())}"
        elif 'shape' in q_lower or 'size' in q_lower:
            answer = f"Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns"
        else:
            answer = f"This is a descriptive query about '{dataset_name}'. The first 3 rows are:\n\n{df.head(3).to_string()}"
        
        return {
            'question': question, 'answer': answer, 'visualization_html': None,
            'visualization_json': None, 'method': 'descriptive_summary'
        }

    def _handle_fallback_analysis(self, question: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Safe fallback when routing/handlers fail."""
        rows, cols = df.shape
        answer = (
            f"Fallback analysis invoked.\n\n"
            f"Dataset shape: {rows:,} rows × {cols} columns\n"
        )
        return {
            'question': question, 'answer': answer, 'visualization_html': None,
            'visualization_json': None, 'method': 'fallback_basic'
        }