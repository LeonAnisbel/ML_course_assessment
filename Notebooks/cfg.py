
c = get_config()
c.NbConvertApp.export_format = 'pdf'
c.TemplateExporter.extra_template_basedirs = ['.']
c.Exporter.template_file = 'my_pdf_template.tex.j2'
