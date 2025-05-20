-- Processes a string of text to find Obsidian-style wikilinks
-- and replace them with their display text.
local function process_text_for_wikilinks(text_str)
  local new_inlines_for_text = pandoc.Inlines {}
  local i = 1 -- Current position in text_str

  while i <= #text_str do
    -- Find the next occurrence of "[["
    local start_link_pos = text_str:find("%[%[", i)

    if not start_link_pos then
      -- No more "[[" found, add the rest of the text and finish
      if i <= #text_str then
        new_inlines_for_text:insert(pandoc.Str(text_str:sub(i)))
      end
      break
    end

    -- Add text before this potential link
    if start_link_pos > i then
      new_inlines_for_text:insert(pandoc.Str(text_str:sub(i, start_link_pos - 1)))
    end

    -- Try to find the matching "]]" for the found "[["
    -- Start searching for "]]" after the "[["
    local end_link_pos = text_str:find("%]%]", start_link_pos + 2)

    if not end_link_pos then
      -- No matching "]]" found. Treat the "[[" as literal text.
      new_inlines_for_text:insert(pandoc.Str(text_str:sub(start_link_pos, start_link_pos + 1))) -- Add the literal "[["
      i = start_link_pos + 2                                                                    -- Continue searching after the literal "[["
    else
      -- Found a complete "[[content]]" structure
      local content = text_str:sub(start_link_pos + 2, end_link_pos - 1)
      local display_name

      -- Check for a pipe "|" within the content to distinguish [[path|name]] from [[name]]
      local pipe_char_pos = content:find("|", 1, true) -- plain find for the first pipe

      if pipe_char_pos then
        -- Link is like [[path|name]] or [[path|]]
        display_name = content:sub(pipe_char_pos + 1)
        -- If display_name is empty (e.g., from [[path|]]), use the part before the pipe
        if #display_name == 0 then
          display_name = content:sub(1, pipe_char_pos - 1)
        end
      else
        -- No pipe, link is like [[name]]
        display_name = content
      end

      new_inlines_for_text:insert(pandoc.Str(display_name))
      i = end_link_pos + 2 -- Continue searching after the processed "]]"
    end
  end
  return new_inlines_for_text
end

-- This is the main filter function Pandoc will call for sequences of inline elements.
function Inlines(inlines_list_input)
  local new_inlines_list_output_parts = {}  -- Using a Lua table to build the new list of inlines
  local current_text_accumulator_parts = {} -- Accumulates text from Str and Space elements

  local function flush_text_accumulator()
    if #current_text_accumulator_parts > 0 then
      local full_accumulated_text = table.concat(current_text_accumulator_parts)
      -- process_text_for_wikilinks returns a pandoc.Inlines object (which is a list of inlines)
      local processed_inlines_from_text = process_text_for_wikilinks(full_accumulated_text)
      for _, inline_element in ipairs(processed_inlines_from_text) do
        table.insert(new_inlines_list_output_parts, inline_element)
      end
      current_text_accumulator_parts = {} -- Reset accumulator
    end
  end

  for _, element in ipairs(inlines_list_input) do
    if element.t == "Str" then
      table.insert(current_text_accumulator_parts, element.text)
    elseif element.t == "Space" then
      table.insert(current_text_accumulator_parts, " ") -- Convert Space element to a space character
    else
      -- This element is not Str or Space (e.g., Emph, Strong, Link, Image).
      -- First, process any text accumulated so far.
      flush_text_accumulator()
      -- Then, add this complex element to our output list.
      -- Pandoc will recursively walk its children; if those children
      -- form an Inlines list, this Inlines filter will be called on them automatically.
      table.insert(new_inlines_list_output_parts, element)
    end
  end

  -- After iterating through all elements, process any remaining accumulated text.
  flush_text_accumulator()

  return pandoc.Inlines(new_inlines_list_output_parts) -- Convert Lua table to a pandoc.Inlines object
end
