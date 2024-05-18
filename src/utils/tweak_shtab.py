# HACK for shtab
# https://github.com/iterative/shtab/issues/65

import shtab
from jsonargparse import ActionConfigFile
from shtab import (
    CHOICE_FUNCTIONS,
    FLAG_OPTION,
    OPTION_END,
    OPTION_MULTI,
    SUPPRESS,
    Choice,
    Template,
    complete2pattern,
    escape_zsh,
    get_public_subcommands,
    log,
    mark_completer,
    starmap,
    wordify,
)

OPTION_MULTI = (*OPTION_MULTI, ActionConfigFile)


@mark_completer("zsh")
def complete_zsh(parser, root_prefix=None, preamble="", choice_functions=None):
    """
    Returns zsh syntax autocompletion script.
    See `complete` for arguments.
    """
    prog = parser.prog
    root_prefix = wordify("_shtab_" + (root_prefix or prog))

    choice_type2fn = {k: v["zsh"] for k, v in CHOICE_FUNCTIONS.items()}
    if choice_functions:
        choice_type2fn.update(choice_functions)

    def format_optional(opt):
        return (
            (
                '{nargs}{options}"[{help}]"'
                if isinstance(opt, FLAG_OPTION)
                else '{nargs}{options}"[{help}]:{dest}:{pattern}"'
            )
            .format(
                nargs=(
                    '"(- :)"'
                    if isinstance(opt, OPTION_END)
                    else '"*"'
                    if isinstance(opt, OPTION_MULTI)
                    else ""
                ),
                options=(
                    "{{{}}}".format(",".join(opt.option_strings))
                    if len(opt.option_strings) > 1
                    else '"{}"'.format("".join(opt.option_strings))
                ),
                help=escape_zsh(opt.help or ""),
                dest=opt.dest,
                pattern=complete2pattern(opt.complete, "zsh", choice_type2fn)
                if hasattr(opt, "complete")
                else (
                    choice_type2fn[opt.choices[0].type]
                    if isinstance(opt.choices[0], Choice)
                    else "({})".format(" ".join(map(str, opt.choices)))
                )
                if opt.choices
                else "_default",
            )
            .replace('""', "")
        )

    def format_positional(opt):
        return '"{nargs}:{help}:{pattern}"'.format(
            nargs={"+": "(*)", "*": "(*):"}.get(opt.nargs, ""),
            help=escape_zsh((opt.help or opt.dest).strip().split("\n")[0]),
            pattern=complete2pattern(opt.complete, "zsh", choice_type2fn)
            if hasattr(opt, "complete")
            else (
                choice_type2fn[opt.choices[0].type]
                if isinstance(opt.choices[0], Choice)
                else "({})".format(" ".join(map(str, opt.choices)))
            )
            if opt.choices
            else "_default",
        )

    # {cmd: {"help": help, "arguments": [arguments]}}
    all_commands = {
        root_prefix: {
            "cmd": prog,
            "arguments": [
                format_optional(opt)
                for opt in parser._get_optional_actions()
                if opt.help != SUPPRESS
            ],
            "help": (parser.description or "").strip().split("\n")[0],
            "commands": [],
            "paths": [],
        }
    }

    def recurse(parser, prefix, paths=None):
        paths = paths or []
        subcmds = []
        for sub in parser._get_positional_actions():
            if sub.help == SUPPRESS or not sub.choices:
                continue
            if not sub.choices or not isinstance(sub.choices, dict):
                # positional argument
                all_commands[prefix]["arguments"].append(format_positional(sub))
            else:  # subparser
                log.debug(f"choices:{prefix}:{sorted(sub.choices)}")
                public_cmds = get_public_subcommands(sub)
                for cmd, subparser in sub.choices.items():
                    if cmd not in public_cmds:
                        log.debug("skip:subcommand:%s", cmd)
                        continue
                    log.debug("subcommand:%s", cmd)

                    # optionals
                    arguments = [
                        format_optional(opt)
                        for opt in subparser._get_optional_actions()
                        if opt.help != SUPPRESS
                    ]

                    # positionals
                    arguments.extend(
                        format_positional(opt)
                        for opt in subparser._get_positional_actions()
                        if not isinstance(opt.choices, dict)
                        if opt.help != SUPPRESS
                    )

                    new_pref = prefix + "_" + wordify(cmd)
                    options = all_commands[new_pref] = {
                        "cmd": cmd,
                        "help": (subparser.description or "").strip().split("\n")[0],
                        "arguments": arguments,
                        "paths": [*paths, cmd],
                    }
                    new_subcmds = recurse(subparser, new_pref, [*paths, cmd])
                    options["commands"] = {
                        all_commands[pref]["cmd"]: all_commands[pref]
                        for pref in new_subcmds
                        if pref in all_commands
                    }
                    subcmds.extend([*new_subcmds, new_pref])
                    log.debug("subcommands:%s:%s", cmd, options)
        return subcmds

    recurse(parser, root_prefix)
    all_commands[root_prefix]["commands"] = {
        options["cmd"]: options
        for prefix, options in sorted(all_commands.items())
        if len(options.get("paths", [])) < 2 and prefix != root_prefix
    }
    subcommands = {
        prefix: options
        for prefix, options in all_commands.items()
        if options.get("commands")
    }
    subcommands.setdefault(root_prefix, all_commands[root_prefix])
    log.debug("subcommands:%s:%s", root_prefix, sorted(all_commands))

    def command_case(prefix, options):
        name = options["cmd"]
        commands = options["commands"]
        case_fmt_on_no_sub = """{name}) _arguments -C ${prefix}_{name}_options ;;"""
        case_fmt_on_sub = """{name}) {prefix}_{name} ;;"""

        cases = []
        for _, options in sorted(commands.items()):
            fmt = case_fmt_on_sub if options.get("commands") else case_fmt_on_no_sub
            cases.append(fmt.format(name=options["cmd"], prefix=prefix))
        cases = "\n\t".expandtabs(8).join(cases)

        return f"""\
{prefix}() {{
  local context state line curcontext="$curcontext"
  _arguments -C ${prefix}_options \\
    ': :{prefix}_commands' \\
    '*::: :->{name}'
  case $state in
    {name})
      words=($line[1] "${{words[@]}}")
      (( CURRENT += 1 ))
      curcontext="${{curcontext%:*:*}}:{prefix}-$line[1]:"
      case $line[1] in
        {cases}
      esac
  esac
}}
"""

    def command_option(prefix, options):
        return """\
{prefix}_options=(
  {arguments}
)
""".format(prefix=prefix, arguments="\n  ".join(options["arguments"]))

    def command_list(prefix, options):
        name = " ".join([prog, *options["paths"]])
        commands = "\n    ".join(
            '"{}:{}"'.format(cmd, escape_zsh(opt["help"]))
            for cmd, opt in sorted(options["commands"].items())
        )
        return f"""
{prefix}_commands() {{
  local _commands=(
    {commands}
  )
  _describe '{name} commands' _commands
}}"""

    preamble = (
        f"""\
# Custom Preamble
{preamble.rstrip()}
# End Custom Preamble
"""
        if preamble
        else ""
    )
    # References:
    #   - https://github.com/zsh-users/zsh-completions
    #   - http://zsh.sourceforge.net/Doc/Release/Completion-System.html
    #   - https://mads-hartmann.com/2017/08/06/
    #     writing-zsh-completion-scripts.html
    #   - http://www.linux-mag.com/id/1106/
    return Template(
        """\
#compdef ${prog}
# AUTOMATCALLY GENERATED by `shtab`
${command_commands}
${command_options}
${command_cases}
${preamble}
typeset -A opt_args
${root_prefix} "$@\""""
    ).safe_substitute(
        prog=prog,
        root_prefix=root_prefix,
        command_cases="\n".join(starmap(command_case, sorted(subcommands.items()))),
        command_commands="\n".join(starmap(command_list, sorted(subcommands.items()))),
        command_options="\n".join(
            starmap(command_option, sorted(all_commands.items()))
        ),
        preamble=preamble,
    )


shtab.complete_zsh = complete_zsh
