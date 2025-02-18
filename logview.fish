#!/usr/bin/env fish

function logview
    argparse --name=logview 'f/fields=' 'F/filter=' 's/sort=' -- $argv
    or return 1

    if test (count $argv) -lt 1
	echo "Usage: logview <log_files> [--fields <field1,field2,...>] [--filter <field=value>]"
	return 1
    end

    for log_file in $argv
	if not test -f $log_file
	    echo "Error: File '$log_file' not found!"
	    return 1
	end
    end

    # Filter JSON logs
    set jq_filter '.'
    if set -q _flag_filter
	set filter_field (echo $_flag_filter | cut -d= -f1)
	set filter_value (echo $_flag_filter | cut -d= -f2-)

	if string match -r '^-?[0-9]+(\.[0-9]+)?$' "$filter_value"
	    set jq_filter "$jq_filter | select( .$filter_field == ($filter_value | tonumber))"
	else
	    set jq_filter "$jq_filter | select( (.$filter_field // \"\") | contains(\"$filter_value\"))"
	end
    end
    
    if set -q _flag_fields
	set jq_filter "$jq_filter | {$_flag_fields}"
    end

    set log_files (string join " " $argv)
    set base_cmd "cat $log_files | jq -c '$jq_filter'"

    if set -q _flag_sort
	set sort_field $_flag_sort
	set final_cmd "$base_cmd | jq -s -c 'sort_by(.$sort_field) | .[]'"
    else
	set final_cmd $base_cmd
    end
    

    # Debugging (uncomment to print the generated jq query)
    echo "DEBUG: jq_filter = $jq_filter"


    eval $final_cmd | fzf \
      --header="Filter logs by timestamp, level, event, message, or data" \
      --delimiter="\t" \
      --preview 'echo {} | jq | batcat --language=json --color=always' \
      --bind "enter:execute(echo {} | jq -C | less -R)"

end
