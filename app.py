import streamlit as st
import wget
import os
import numpy as np
import torch
import faiss
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForCausalLM
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="RAG vs LLM Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGSystem:
    """
    Retrieval-Augmented Generation System using DPR and GPT-2
    
    This system demonstrates the difference between:
    1. Plain LLM responses (without context)
    2. RAG responses (with retrieved relevant context)
    """
    
    def __init__(self):
        self.context_encoder = None
        self.context_tokenizer = None
        self.question_encoder = None
        self.question_tokenizer = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.index = None
        self.paragraphs = []
        self.context_embeddings = None
    
    def tsne_plot(self, data, labels=None, title="3D t-SNE Visualization of Embeddings"):
        """Create 3D t-SNE visualization of embeddings"""
        # Apply t-SNE to reduce to 3D
        perplexity = min(15, data.shape[0] - 1)  # Smaller perplexity for small datasets
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity, n_iter=500)
        data_3d = tsne.fit_transform(data)
        
        # Create the plot with smaller size
        fig = plt.figure(figsize=(10, 6))  # Smaller figure size
        ax = fig.add_subplot(111, projection='3d')
        
        # Define distinct colors and markers
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Plot each point with distinct styling
        for idx, point in enumerate(data_3d):
            if labels:
                # Determine point type and styling
                if "Your Question" in labels[idx]:
                    color = 'red'
                    marker = 'o'
                    size = 100
                    alpha = 1.0
                elif "Context" in labels[idx] and "Other" not in labels[idx]:
                    color = 'blue'
                    marker = 's'
                    size = 80
                    alpha = 0.8
                else:  # Other contexts
                    color = 'lightgray'
                    marker = '^'
                    size = 40
                    alpha = 0.6
                
                ax.scatter(point[0], point[1], point[2], 
                          color=color, marker=marker, s=size, alpha=alpha)
            else:
                ax.scatter(point[0], point[1], point[2], 
                          color=colors[idx % len(colors)], s=60)
        
        # Adding labels and titles
        ax.set_xlabel('t-SNE Component 1', fontsize=10)
        ax.set_ylabel('t-SNE Component 2', fontsize=10)
        ax.set_zlabel('t-SNE Component 3', fontsize=10)
        ax.set_title(title, fontsize=12)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='❓ Your Question'),
            Patch(facecolor='blue', label='📄 Retrieved Contexts'),  
            Patch(facecolor='lightgray', label='📋 Other Contexts')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        return fig
    
    @st.cache_resource
    def load_models(_self):
        """Load all required models and tokenizers"""
        try:
            # Load DPR context encoder
            _self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                'facebook/dpr-ctx_encoder-single-nq-base'
            )
            _self.context_encoder = DPRContextEncoder.from_pretrained(
                'facebook/dpr-ctx_encoder-single-nq-base'
            )
            
            # Load DPR question encoder
            _self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                'facebook/dpr-question_encoder-single-nq-base'
            )
            _self.question_encoder = DPRQuestionEncoder.from_pretrained(
                'facebook/dpr-question_encoder-single-nq-base'
            )
            
            # Load GPT-2 for text generation
            _self.llm_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            _self.llm_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
            _self.llm_model.generation_config.pad_token_id = _self.llm_tokenizer.pad_token_id
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def download_and_process_data(self, url, filename):
        """Download and process company policies document"""
        try:
            # Download file if it doesn't exist
            if not os.path.exists(filename):
                with st.spinner("Downloading company policies document..."):
                    wget.download(url, out=filename)
            
            # Read and split text into paragraphs
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Split into paragraphs and filter empty ones
            paragraphs = text.split('\n')
            self.paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 10]
            
            return True
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return False
    
    def create_embeddings(self):
        """Create embeddings for all paragraphs and build FAISS index"""
        try:
            with st.spinner("Creating embeddings for document chunks..."):
                embeddings = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(self.paragraphs):
                    inputs = self.context_tokenizer(
                        text, return_tensors='pt', padding=True, 
                        truncation=True, max_length=256
                    )
                    outputs = self.context_encoder(**inputs)
                    embeddings.append(outputs.pooler_output)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(self.paragraphs))
                
                # Convert to numpy array
                self.context_embeddings = torch.cat(embeddings).detach().numpy().astype('float32')
                
                # Create FAISS index
                embedding_dim = self.context_embeddings.shape[1]
                self.index = faiss.IndexFlatL2(embedding_dim)
                self.index.add(self.context_embeddings)
                
                progress_bar.empty()
                st.success(f"Successfully processed {len(self.paragraphs)} document chunks!")
                return True
                
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def search_relevant_contexts(self, question, k=5):
        """Search for most relevant contexts to the question"""
        try:
            # Encode the question
            question_inputs = self.question_tokenizer(question, return_tensors='pt')
            question_embedding = self.question_encoder(**question_inputs).pooler_output.detach().numpy()
            
            # Search the index
            distances, indices = self.index.search(question_embedding, k)
            
            # Get the relevant contexts
            relevant_contexts = [self.paragraphs[idx] for idx in indices[0]]
            
            return relevant_contexts, distances[0], indices[0]
        except Exception as e:
            st.error(f"Error searching contexts: {str(e)}")
            return [], [], []
    
    def generate_llm_response(self, question):
        """Generate response using only LLM (no context)"""
        try:
            inputs = self.llm_tokenizer(
                question, return_tensors='pt', max_length=512, truncation=True
            )
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs['input_ids'], 
                    max_length=inputs['input_ids'].shape[1] + 40,  # Shorter responses
                    min_length=inputs['input_ids'].shape[1] + 15,
                    length_penalty=2.0,
                    num_beams=4, 
                    early_stopping=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
            
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input question from the response
            response = response[len(question):].strip()
            
            # Clean up and stop at first complete sentence
            response = self._clean_response(response)
            
            return response if response else "I don't have enough information to answer this question."
            
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"
    
    def generate_rag_response(self, question, contexts):
        """Generate response using RAG (question + retrieved contexts)"""
        try:
            # Combine question with retrieved contexts
            context_text = ' '.join(contexts[:2])  # Use top 2 contexts only
            input_text = f"Context: {context_text}\n\nBased on the above context, answer: {question}\n\nAnswer:"
            
            inputs = self.llm_tokenizer(
                input_text, return_tensors='pt', max_length=1024, truncation=True
            )
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs['input_ids'], 
                    max_length=inputs['input_ids'].shape[1] + 60,  # Shorter responses
                    min_length=inputs['input_ids'].shape[1] + 20,
                    length_penalty=1.5,
                    num_beams=4, 
                    early_stopping=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.6,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3
                )
            
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            # Clean the response
            response = self._clean_response(response)
            
            return response if response else "Based on the company policy, I found relevant information but couldn't generate a clear answer."
            
        except Exception as e:
            return f"Error generating RAG response: {str(e)}"
    
    def _clean_response(self, response):
        """Clean and truncate response to avoid continuing questions"""
        if not response:
            return response
            
        # Remove common artifacts
        artifacts_to_remove = ["Context:", "Question:", "Answer:", "Based on the above context,"]
        for artifact in artifacts_to_remove:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
        
        # Split into sentences and take only complete statements
        sentences = response.split('.')
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Stop if we encounter a question
            if '?' in sentence:
                break
                
            # Stop if sentence is too short or seems incomplete
            if len(sentence) < 10:
                continue
                
            clean_sentences.append(sentence)
            
            # Stop after 2-3 good sentences to keep responses concise
            if len(clean_sentences) >= 2:
                break
        
        return '. '.join(clean_sentences) + '.' if clean_sentences else response

def main():
    st.title("🤖 RAG vs LLM Comparison System")
    st.markdown("""
    This demo compares responses from:
    - **Plain LLM**: GPT-2 generating answers without any context
    - **RAG System**: GPT-2 enhanced with retrieved relevant company policy context
    
    The system uses **DPR (Dense Passage Retrieval)** for finding relevant context and **FAISS** for efficient similarity search.
    """)
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        st.session_state.models_loaded = False
        st.session_state.data_processed = False
    
    rag_system = st.session_state.rag_system
    
    # Sidebar for system setup
    with st.sidebar:
        st.header("🔧 System Setup")
        
        # Step 1: Load Models
        if not st.session_state.models_loaded:
            if st.button("🔄 Load Models", type="primary"):
                with st.spinner("Loading DPR and GPT-2 models..."):
                    if rag_system.load_models():
                        st.session_state.models_loaded = True
                        st.success("✅ Models loaded successfully!")
                        st.rerun()
        else:
            st.success("✅ Models loaded")
        
        # Step 2: Process Data
        if st.session_state.models_loaded and not st.session_state.data_processed:
            if st.button("📁 Process Company Policies", type="primary"):
                url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
                filename = 'companyPolicies.txt'
                
                if rag_system.download_and_process_data(url, filename):
                    if rag_system.create_embeddings():
                        st.session_state.data_processed = True
                        st.rerun()
        elif st.session_state.data_processed:
            st.success("✅ Data processed")
            st.info(f"📊 {len(rag_system.paragraphs)} document chunks loaded")
        
        # System Status
        st.header("📈 System Status")
        st.write("**Models:**", "✅ Ready" if st.session_state.models_loaded else "❌ Not loaded")
        st.write("**Data:**", "✅ Ready" if st.session_state.data_processed else "❌ Not processed")
    
    # Main interface
    if st.session_state.models_loaded and st.session_state.data_processed:
        st.header("💬 Ask Questions About Company Policies")
        
        # Sample questions based on actual company policies document
        sample_questions = [
            "What are the fundamental principles in our Code of Conduct?",
            "What is our policy on discrimination and equal opportunity?", 
            "Can I use company email for personal communications?",
            "What should I do if my mobile phone is lost or stolen?",
            "Where am I allowed to smoke on company premises?",
            "What substances are prohibited in the workplace?",
            "How do I report harassment or discrimination?",
            "What happens if I violate company policies?",
            "What are the security requirements for internet usage?"
        ]
        
        selected_question = st.selectbox(
            "Choose a sample question or type your own:",
            [""] + sample_questions
        )
        
        # User input
        user_question = st.text_input(
            "Your Question:",
            value=selected_question,
            placeholder="Type your question about company policies..."
        )
        
        if user_question and st.button("🔍 Get Answers", type="primary"):
            with st.spinner("Processing your question..."):
                # Search for relevant contexts
                contexts, distances, indices = rag_system.search_relevant_contexts(user_question)
                
                if contexts:
                    # Generate both responses
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🤖 Plain LLM Response")
                        st.caption("GPT-2 without any context")
                        
                        llm_response = rag_system.generate_llm_response(user_question)
                        st.write(llm_response)
                        
                        # Analysis
                        st.info("**Analysis:** This response is generated purely from the model's training data without any company-specific context.")
                    
                    with col2:
                        st.subheader("🧠 RAG System Response")
                        st.caption("GPT-2 enhanced with company policy context")
                        
                        rag_response = rag_system.generate_rag_response(user_question, contexts)
                        st.write(rag_response)
                        
                        # Analysis
                        st.success("**Analysis:** This response uses relevant company policy context to provide specific, accurate information.")
                    
                    # 3D Visualization of Embeddings
                    st.header("🎯 3D Embedding Space Visualization")
                    st.markdown("This visualization shows how your question relates to different document chunks in the embedding space.")
                    
                    # Get question embedding for visualization
                    question_inputs = rag_system.question_tokenizer(user_question, return_tensors='pt')
                    question_embedding = rag_system.question_encoder(**question_inputs).pooler_output.detach().numpy()
                    
                    # Combine question embedding with retrieved context embeddings
                    viz_embeddings = []
                    viz_labels = []
                    
                    # Add question embedding
                    viz_embeddings.append(question_embedding.flatten())
                    viz_labels.append(f"❓ Your Question: {user_question[:40]}...")
                    
                    # Add top retrieved context embeddings (ensure they match the displayed contexts)
                    displayed_contexts = []
                    for i, idx in enumerate(indices[:5]):
                        viz_embeddings.append(rag_system.context_embeddings[idx])
                        # Use the actual context that will be displayed
                        context_text = contexts[i]
                        displayed_contexts.append((context_text, distances[i], idx))
                        
                        # Get policy name from context
                        policy_name = "Unknown Policy"
                        if "Code of Conduct" in context_text:
                            policy_name = "Code of Conduct"
                        elif "Internet and Email" in context_text:
                            policy_name = "Internet & Email Policy"
                        elif "Mobile Phone" in context_text:
                            policy_name = "Mobile Phone Policy"
                        elif "Smoking" in context_text:
                            policy_name = "Smoking Policy"
                        elif "Drug and Alcohol" in context_text:
                            policy_name = "Drug & Alcohol Policy"
                        elif "Health and Safety" in context_text:
                            policy_name = "Health & Safety Policy"
                        elif "Anti-discrimination" in context_text:
                            policy_name = "Anti-discrimination Policy"
                        elif "Discipline and Termination" in context_text:
                            policy_name = "Discipline & Termination Policy"
                        elif "Recruitment" in context_text:
                            policy_name = "Recruitment Policy"
                        
                        viz_labels.append(f"📄 Context {i+1}: {policy_name}")
                    
                    # Add fewer random contexts for cleaner visualization
                    available_indices = [i for i in range(len(rag_system.context_embeddings)) if i not in indices[:5]]
                    if available_indices:
                        random_indices = np.random.choice(
                            available_indices, 
                            size=min(3, len(available_indices)), 
                            replace=False
                        )
                        for i, idx in enumerate(random_indices):
                            viz_embeddings.append(rag_system.context_embeddings[idx])
                            viz_labels.append(f"📋 Other: Policy {idx}")
                    
                    # Create and display the 3D plot
                    viz_data = np.array(viz_embeddings)
                    
                    try:
                        fig = rag_system.tsne_plot(
                            viz_data, 
                            labels=viz_labels,
                            title="3D t-SNE: Question vs Retrieved vs Other Contexts"
                        )
                        st.pyplot(fig)
                        
                        st.info("""
                        **How to interpret this visualization:**
                        - **🔴 Red circles** = Your question in the embedding space
                        - **🔵 Blue squares** = Retrieved relevant contexts (should be closer to your question)
                        - **⚪ Gray triangles** = Random other contexts (should be farther away)
                        - **Closer points** = More semantically similar content
                        """)
                        
                    except Exception as e:
                        st.warning(f"Could not generate visualization: {str(e)}")
                        st.info("This might happen with very small datasets or when embeddings are too similar.")
                    
                    # Show retrieved contexts (use the same contexts as in visualization)
                    st.header("📄 Retrieved Context (Top 5 Most Relevant)")
                    
                    for i, (context, distance, idx) in enumerate(displayed_contexts):
                        similarity_score = 1/(1+distance)
                        with st.expander(f"Context {i+1} - Similarity: {similarity_score:.3f} (Distance: {distance:.4f})"):
                            st.write(context)
                            st.caption(f"Source: Document paragraph {idx}")
                    
                    # Comparison metrics
                    st.header("📊 System Insights")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Contexts Retrieved", len(contexts))
                    with col2:
                        st.metric("Best Match Score", f"{1/(1+distances[0]):.3f}")
                    with col3:
                        st.metric("Worst Match Score", f"{1/(1+distances[-1]):.3f}")
                    with col4:
                        st.metric("Processing Model", "DPR + GPT-2")
    
    else:
        st.warning("⚠️ Please load models and process data using the sidebar controls first.")
        
        # Instructions
        st.header("🚀 Getting Started")
        st.markdown("""
        1. **Load Models**: Click 'Load Models' in the sidebar to initialize DPR and GPT-2
        2. **Process Data**: Click 'Process Company Policies' to download and embed the document
        3. **Ask Questions**: Once setup is complete, ask questions about company policies
        
        ### What This Demo Shows:
        - **Retrieval Quality**: How well the system finds relevant information
        - **Context Impact**: Difference between generic vs. contextual responses  
        - **RAG Architecture**: Complete pipeline from query to answer
        
        ### Technical Components:
        - **DPR (Dense Passage Retrieval)**: For encoding questions and contexts
        - **FAISS**: For fast similarity search across document embeddings
        - **GPT-2**: For natural language generation
        - **Company Policies**: Real document demonstrating enterprise use case
        """)

if __name__ == "__main__":
    main()