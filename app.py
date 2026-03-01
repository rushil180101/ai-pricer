import os
import threading
import time
import gradio as gr
from agents.autonomous_planning_agent import AutonomousPlanningAgent
from typing import Generator
from functools import partial

# List of log files to monitor
LOG_FILES = [
    "autonomous_planning_agent.log",
    "messaging_agent.log",
    "scanner_agent.log",
    "ensemble_agent.log",
    "rag_pipeline_handler.log",
]


def run_with_logs(agent: AutonomousPlanningAgent) -> Generator:
    """
    Generator function that:
    - Streams logs in real time
    - Returns final deals table as a string
    """

    final_result = {"data": None}

    # Run the agent in a background thread
    def run_agent():
        final_result["data"] = agent.execute()  # returns list of tuples

    thread = threading.Thread(target=run_agent)
    thread.start()

    log_positions = {log: 0 for log in LOG_FILES}
    log_contents = {log: "" for log in LOG_FILES}

    # Clear old logs at start
    for log in LOG_FILES:
        if os.path.exists(log):
            open(log, "w").close()

    while thread.is_alive():
        # Collect logs for each file
        for log in LOG_FILES:
            if os.path.exists(log):
                with open(log, "r") as f:
                    f.seek(log_positions[log])
                    new_data = f.read()
                    log_positions[log] = f.tell()

                    if new_data:
                        if log_contents[log]:
                            log_contents[log] += "\n" + new_data
                        else:
                            log_contents[log] = new_data

        # Yield logs for Textboxes, deals = empty while running
        yield *[log_contents[log] for log in LOG_FILES], ""

        time.sleep(1)

    # Final logs after agent finishes
    for log in LOG_FILES:
        if os.path.exists(log):
            with open(log, "r") as f:
                log_contents[log] = f.read()

    # Prepare final deals table as string
    deals_text = ""
    rows = final_result["data"]  # list of tuples
    if rows:
        for product, est_price, deal_price in rows:
            # Split description and keep prices separate
            lines = product.strip().split("\n")
            deals_text += "\n".join(lines) + "\n"
            deals_text += (
                f"Estimated Price: ${est_price} | Deal Price: ${deal_price}\n\n"
            )
    else:
        deals_text = "No deals found."

    # Final yield: logs + final deals string
    yield *[log_contents[log] for log in LOG_FILES], deals_text


if __name__ == "__main__":

    autonomous_planning_agent = AutonomousPlanningAgent()

    # Custom CSS for smaller font, centered title
    custom_css = """
    /* Reduce overall font size */
    .gradio-container {
        font-size: 13px !important;
    }

    /* Center title */
    h1 {
        text-align: center !important;
        font-size: 20px !important;
    }

    /* Textboxes (logs + deals) use monospace */
    textarea {
        font-family: monospace !important;
        font-size: 11px !important;
    }
    """

    with gr.Blocks(css=custom_css) as app:

        gr.Markdown("# 🛍️ Product Deals Finder + AI Pricer")

        run_button = gr.Button("Find Deals")

        # Logs textboxes row
        log_boxes = []
        with gr.Row():
            for lf in LOG_FILES:
                log_boxes.append(
                    gr.Textbox(
                        label=lf.replace(".log", ""),
                        lines=20,
                        interactive=False,
                    )
                )

        # Deals textbox below logs
        deals_box = gr.Textbox(
            label="Deals found",
            lines=15,
            interactive=False,
        )

        # Connect button to generator function
        run_button.click(
            fn=partial(run_with_logs, autonomous_planning_agent),
            inputs=[],
            outputs=log_boxes + [deals_box],
        )

    print("Launching AI Pricer App")
    app.launch(inbrowser=True)
