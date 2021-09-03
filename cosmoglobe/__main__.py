import click
from cosmoglobe.plot.click_plotting import commands_plotting

CONTEXT_SETTINGS = dict(max_content_width=100)

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    pass

cli = click.CommandCollection(sources=[commands_plotting,], context_settings=CONTEXT_SETTINGS)

if __name__ == '__main__':
    cli()
    