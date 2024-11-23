import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from typing import Dict, List
import time
import json
import glob

# Import algorithms
from simulated_annealing import SimulatedAnnealingScheduler
from genetic_algorithm import GeneticScheduler
from greedy import FastGreedyScheduler
from scipy_optimize import OptimalTaskScheduler

# Page configuration
st.set_page_config(
    page_title="Algorithm Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title('Task Scheduling Algorithm Comparison')
st.markdown("""
Compare different scheduling algorithms performance on various dataset sizes.
Select algorithms and dataset size from the sidebar to begin comparison.
""")

# Define constants
ALGORITHMS = {
    'Simulated Annealing': SimulatedAnnealingScheduler,
    'Genetic Algorithm': GeneticScheduler,
    'Greedy Algorithm': FastGreedyScheduler,
    'Scipy Optimize': OptimalTaskScheduler
}

DATASETS = {
    'Small': 'advanced_task_scheduling_small_dataset.json',
    'Medium': 'advanced_task_scheduling_medium_dataset.json',
    'Large': 'advanced_task_scheduling_large_dataset.json'
}


def get_makespan(results: Dict) -> float:
    """Extract makespan from algorithm results"""
    if not results:
        return 0.0

    # Try different possible locations of makespan
    if 'makespan' in results:
        return float(results['makespan'])
    elif 'best_cost' in results:
        return float(results['best_cost'])
    elif 'optimization_results' in results and 'makespan' in results['optimization_results']:
        return float(results['optimization_results']['makespan'])
    elif 'schedule' in results:
        schedule = results['schedule']
        if isinstance(schedule, list) and schedule and 'end_time' in schedule[-1]:
            return float(max(task['end_time'] for task in schedule))

    st.warning(f"Could not extract makespan from results with keys: {results.keys()}")
    return 0.0


def extract_metrics_from_report(report: Dict) -> Dict:
    """Extract metrics from report with proper type handling"""
    metrics = {
        'Algorithm': '',
        'Makespan': 0.0,
        'Execution Time': 0.0,
        'Deadline Violations': 0,
        'Resource Violations': 0
    }

    try:
        # Special handling for SA results
        if 'performance_metrics' in report:
            perf = report['performance_metrics']
            violations = perf.get('violations', {})

            metrics.update({
                'Makespan': float(perf.get('makespan', 0.0)),
                'Execution Time': float(perf.get('execution_time', 0.0)),
                'Deadline Violations': int(violations.get('deadline', 0)),
                'Resource Violations': int(violations.get('resource', 0))
            })

            # Verify values are reasonable
            if metrics['Makespan'] < 0 or metrics['Makespan'] == float('inf'):
                st.warning("Invalid makespan value detected")
                metrics['Makespan'] = 0.0

            if metrics['Resource Violations'] < 0:
                st.warning("Invalid resource violations value detected")
                metrics['Resource Violations'] = 0

        else:
            # Handle legacy format
            violations = report.get('violations', {})
            metrics.update({
                'Makespan': float(report.get('makespan', 0.0)),
                'Execution Time': float(report.get('execution_time', 0.0)),
                'Deadline Violations': int(violations.get('deadline', 0)),
                'Resource Violations': int(violations.get('resource', 0))
            })

        return metrics

    except Exception as e:
        st.error(f"Error extracting metrics: {str(e)}")
        return metrics


def create_visualizations(df: pd.DataFrame):
    """Create visualizations with proper error handling and comprehensive metrics"""
    try:
        # Ensure all required columns exist and validate data
        required_columns = ['Algorithm', 'Makespan', 'Execution Time', 'Deadline Violations', 'Resource Violations']
        if not all(col in df.columns for col in required_columns):
            st.error("Missing required columns in data")
            st.write("Available columns:", df.columns.tolist())
            return

        # Data cleanup and validation
        for col in ['Makespan', 'Execution Time', 'Deadline Violations', 'Resource Violations']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0)

        # Create tabs with comprehensive metrics
        tabs = st.tabs(['📊 Performance Comparison', '📈 Violations Analysis', '📋 Table Data'])

        with tabs[0]:
            # Performance Metrics Visualization
            col1, col2 = st.columns(2)

            with col1:
                # Makespan comparison
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                x = range(len(df))
                bars = ax1.bar(x, df['Makespan'], color='skyblue')

                # Customize plot
                ax1.set_xticks(x)
                ax1.set_xticklabels(df['Algorithm'], rotation=45)
                ax1.set_title('Makespan Comparison')
                ax1.grid(True, alpha=0.3)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:,.0f}',
                             ha='center', va='bottom')

                st.pyplot(fig1)

            with col2:
                # Execution time comparison
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                bars = ax2.bar(x, df['Execution Time'], color='lightgreen')

                # Customize plot
                ax2.set_xticks(x)
                ax2.set_xticklabels(df['Algorithm'], rotation=45)
                ax2.set_title('Execution Time (seconds)')
                ax2.grid(True, alpha=0.3)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.2f}s',
                             ha='center', va='bottom')

                st.pyplot(fig2)

        with tabs[1]:
            # Violations Analysis
            col1, col2 = st.columns(2)

            with col1:
                # Deadline violations
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                bars = ax3.bar(x, df['Deadline Violations'], color='salmon')

                ax3.set_xticks(x)
                ax3.set_xticklabels(df['Algorithm'], rotation=45)
                ax3.set_title('Deadline Violations')
                ax3.grid(True, alpha=0.3)

                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:,.0f}',
                             ha='center', va='bottom')

                st.pyplot(fig3)

            with col2:
                # Resource violations
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                bars = ax4.bar(x, df['Resource Violations'], color='plum')

                ax4.set_xticks(x)
                ax4.set_xticklabels(df['Algorithm'], rotation=45)
                ax4.set_title('Resource Violations')
                ax4.grid(True, alpha=0.3)

                for bar in bars:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:,.0f}',
                             ha='center', va='bottom')

                st.pyplot(fig4)

            # Add violations percentage analysis
            st.subheader("Violations Analysis")
            total_tasks = len(df)
            metrics_df = pd.DataFrame({
                'Algorithm': df['Algorithm'],
                'Deadline Violations %': (df['Deadline Violations'] / total_tasks * 100).round(2),
                'Resource Violations %': (df['Resource Violations'] / (total_tasks * 3) * 100).round(2)
                # Assuming 3 resources
            })
            st.dataframe(metrics_df.style.highlight_min(['Deadline Violations %', 'Resource Violations %']))

        # Raw data tab with enhanced display
        with tabs[2]:
            st.subheader("Detailed Results")

            # Format metrics for display
            display_df = pd.DataFrame({
                'Algorithm': df['Algorithm'],
                'Makespan': df['Makespan'].round(2),
                'Execution Time (s)': df['Execution Time'].round(2),
                'Deadline Violations': df['Deadline Violations'],
                'Resource Violations': df['Resource Violations'],
                'Total Violations': df['Deadline Violations'] + df['Resource Violations']
            })

            # Add styling
            styled_df = display_df.style \
                .highlight_min(['Makespan', 'Execution Time (s)', 'Total Violations']) \
                .format({
                'Makespan': '{:.2f}',
                'Execution Time (s)': '{:.2f}',
                'Deadline Violations': '{:,.0f}',
                'Resource Violations': '{:,.0f}',
                'Total Violations': '{:,.0f}'
            })

            st.dataframe(styled_df)

            # Download option with formatted data
            csv = display_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                "algorithm_comparison_results.csv",
                "text/csv",
                key='download-csv'
            )

    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")
        st.write("Data shape:", df.shape)
        st.write("Data columns:", df.columns.tolist())
        st.write("Data head:", df.head())
        st.exception(e)

def plot_comparison(df: pd.DataFrame, metric: str, title: str) -> plt.Figure:
    """Create standardized comparison plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    bars = ax.bar(x, df[metric])

    ax.set_xticks(x)
    ax.set_xticklabels(df['Algorithm'], rotation=45)
    ax.set_title(title)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if metric == 'Execution Time':
            label = f'{height:.2f}s'
        else:
            label = f'{height:,.0f}'
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                label, ha='center', va='bottom')

    plt.tight_layout()
    return fig

def run_algorithms(selected_algos: List[str], dataset_size: str) -> pd.DataFrame:
    """Run algorithms and collect results"""
    results = []
    dataset_path = os.path.join('advanced_task_scheduling_datasets', DATASETS[dataset_size])

    for algo_name in selected_algos:
        with st.container():
            status = st.empty()
            status.write(f"Running {algo_name}...")

            try:
                # Initialize and run algorithm
                algo_class = ALGORITHMS[algo_name]
                scheduler = algo_class(dataset_path)
                algo_results = scheduler.optimize()

                # Extract metrics
                metrics = extract_metrics_from_report(algo_results)
                metrics['Algorithm'] = algo_name
                results.append(metrics)

                status.success(f"✅ {algo_name} completed successfully!")

            except Exception as e:
                status.error(f"❌ Error running {algo_name}: {str(e)}")
                st.exception(e)

    # Create DataFrame and ensure types
    df = pd.DataFrame(results)
    if not df.empty:
        df['Makespan'] = pd.to_numeric(df['Makespan'], errors='coerce')
        df['Execution Time'] = pd.to_numeric(df['Execution Time'], errors='coerce')
        df['Deadline Violations'] = pd.to_numeric(df['Deadline Violations'], errors='coerce')
        df['Resource Violations'] = pd.to_numeric(df['Resource Violations'], errors='coerce')

    return df

def read_algorithm_report(algo_name: str) -> Dict:
    """Read and validate report file"""
    report_patterns = {
        'Simulated Annealing': 'sa_results_*/analysis_report.json',  # Confirm this matches SA output
        'Genetic Algorithm': 'genetic_results_*/analysis_report.json',
        'Greedy Algorithm': 'greedy_results_*/greedy_report.json',
        'Scipy Optimize': 'optimal_results_*/optimal_report.json'
    }

    try:
        pattern = report_patterns[algo_name]
        # Get absolute path to project directory
        base_dir = os.getcwd()
        report_files = sorted(glob.glob(os.path.join(base_dir, pattern)), reverse=True)

        if not report_files:
            st.warning(f"No report file found for {algo_name}")
            return None

        with open(report_files[0], 'r') as f:
            report = json.load(f)

        # Add more detailed validation for SA results
        if algo_name == 'Simulated Annealing':
            if 'performance_metrics' not in report:
                st.warning(f"Invalid SA report format - missing performance metrics")
                return None
            required_metrics = ['makespan', 'execution_time', 'violations']
            if not all(metric in report['performance_metrics'] for metric in required_metrics):
                st.warning(f"Invalid SA report format - missing required metrics")
                return None

        return report

    except Exception as e:
        st.error(f"Error reading report for {algo_name}: {str(e)}")
        return None

def main():
    # Sidebar configuration
    with st.sidebar:
        st.title("Settings")

        # Algorithm selection
        st.subheader("1. Select Algorithms")
        selected_algos = st.multiselect(
            "Choose algorithms",
            options=list(ALGORITHMS.keys()),
            default=list(ALGORITHMS.keys())[:2]
        )

        # Dataset selection
        st.subheader("2. Select Dataset")
        dataset_size = st.radio(
            "Choose dataset size",
            options=list(DATASETS.keys())
        )

        # Run button
        run_button = st.button('🚀 Run Comparison')

        # Help
        with st.expander("ℹ️ Help"):
            st.markdown("""
            1. Select 2+ algorithms
            2. Choose dataset size
            3. Click Run Comparison
            4. View results in tabs
            5. Download if needed
            """)

    # Main flow
    if len(selected_algos) < 2:
        st.warning("⚠️ Please select at least 2 algorithms")
    elif run_button:
        with st.spinner("Running comparison..."):
            results_df = run_algorithms(selected_algos, dataset_size)

            if not results_df.empty:
                create_visualizations(results_df)
                st.success("✅ Comparison completed!")
            else:
                st.error("❌ No results to compare")


if __name__ == "__main__":
    main()
