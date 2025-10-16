import gradio as gr
import pandas as pd
from scripts.theme_load import get_theme
from scripts.savefileunparsed import save_files_unparsed
from scripts.parse_resume_Gemini import runjob

theme = get_theme()

currdata = pd.read_csv("src/mdData.csv").sort_values(by='candidate_score', ascending=False).reset_index(drop = True).to_html(render_links=True, escape=True)

def refreshhtmlfunc():
    return pd.read_csv("src/mdData.csv").sort_values(by='candidate_score', ascending=False).reset_index(drop = True).to_html(render_links=True, escape=True)

# Main page
with gr.Blocks(theme, title="SmartResumeScreener") as demo:
    gr.Navbar(visible=True, main_page_name="Dashboard")
    gr.Markdown(value='''# <center>Welcome to SmartResumeScreener</center>
## <center>Upload all resumes together, enter the job description, and find the eligible candidates by their matching scores</center>
<center>Get started, select a way:</center>''')
    
    gr.Button(value="Use Gemini API", link="geminiapi")
    gr.Button(value="View Parsed Resumes",link="ViewParsed")


# Gemini API page
with demo.route(name="Use Gemini API", path="geminiapi", show_in_navbar=True):
    with gr.Row():
        with gr.Column():
            fileupload = gr.File(label="Upload 1 or more Resume",file_count="multiple",file_types=[".pdf", ".txt", ".rtf", ".docx"])
            fileuploadoutput = gr.Textbox(label="Saved File Paths", lines=5)
        JDbox = gr.Textbox(label="Job Description", placeholder="Enter Job Description", max_lines=12,)
    calcfitnessbutton = gr.Button(value="Calculate resume fitness scores")
    gr.Button(value="Go to Home Page", link="/")
    
    # logic
    fileupload.upload(show_progress='full', fn=save_files_unparsed, inputs=[fileupload], outputs=[fileuploadoutput])
    calcfitnessbutton.click(fn=runjob, inputs=[JDbox], outputs=[fileuploadoutput])

with demo.route(name="View Parsed Resumes", path="ViewParsed", show_in_navbar=True):
    gr.Button("Back to Main", link="/")
    refrshhtml = gr.Button("Refresh")
    htmldisplay = gr.HTML(value = currdata)
    
    refrshhtml.click(fn=refreshhtmlfunc, outputs=[htmldisplay])
    

demo.launch()