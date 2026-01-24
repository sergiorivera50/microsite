import markdown
import re
from pathlib import Path
from datetime import datetime
import tomli
import shutil

# ===============
# Basic Utilities
# ===============

def slugify(text):
    """Convert text to URL-friendly slug"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = re.sub(r'^\-|\-$', '', text)
    return text

def split_content(html, delimiter='<!-- split -->'):
    """Split homepage content via custom delimiter"""
    parts = html.split(delimiter)
    return {
        'intro': parts[0].strip() if len(parts) > 0 else '',
        'rest': parts[1].strip() if len(parts) > 1 else ''
    }

def calculate_reading_time(content, words_per_minute=200):
    """Calculate estimated reading time in minutes"""
    # Strip LaTeX blocks
    text = re.sub(r'\$\$.*?\$\$', '', content, flags=re.DOTALL)  # block math
    text = re.sub(r'\$.*?\$', '', text)  # inline math
    
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Count words
    words = len(text.split())
    
    # Calculate minutes, minimum 1
    minutes = max(1, round(words / words_per_minute))
    
    return minutes

def load_config(file_path):
    """Load configuration from a TOML file"""
    try:
        with open(file_path, "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        raise Exception(f"Config file '{file_path}' not found")

def ensure_dir(directory):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def copy_files(source_dir, target_dir, pattern="**/*", exclude_patterns=None):
    """Generic file copy utility"""
    exclude_patterns = exclude_patterns or []
    source_path = Path(source_dir)

    if not source_path.exists():
        return False

    for item in source_path.glob(pattern):
        # Skip directories
        if item.is_dir():
            continue

        # Skip excluded patterns
        if any(item.match(pat) for pat in exclude_patterns):
            continue

        # Determine destination path
        rel_path = item.relative_to(source_path)
        dest_path = Path(target_dir) / rel_path

        # Create parent directories
        ensure_dir(dest_path.parent)

        # Copy the file
        shutil.copy2(item, dest_path)

    return True

# ==================
# Content Processing
# ==================

def generate_url(rel_path, output_dir, is_index, is_content_index, slug):
    """Determine the URL and output path for a content file"""
    if is_index:
        if rel_path == Path(''):  # root _index.md
            return "/", Path(output_dir) / "index.html"
        else:  # section _index.md
            return f"/{rel_path}/", Path(output_dir) / rel_path / "index.html"
    elif is_content_index:  # content index.md
        return f"/{rel_path}/", Path(output_dir) / rel_path / "index.html"
    else:
        if rel_path == Path(''):  # top-level page
            return f"/{slug}/", Path(output_dir) / slug / "index.html"
        else:  # nested page
            return f"/{rel_path}/{slug}/", Path(output_dir) / rel_path / slug / "index.html"

def render_markdown(text):
    """
    Render markdown text to HTML with custom handling for:
    1. LaTeX blocks (preserved for later JS rendering)
    2. Custom image syntax with size specifications: ![alt|width](src)
    """

    # Store LaTeX parts
    placeholders = {}
    count = 0

    # Handle block LaTeX ($$...$$)
    def replace_block(match):
        nonlocal count
        placeholder = f"LATEX_BLOCK_{count}_"
        placeholders[placeholder] = match.group(0)
        count += 1
        return placeholder

    text = re.sub(r'(?<!\\)\$\$(.*?)(?<!\\)\$\$', replace_block, text, flags=re.DOTALL)

    # Handle inline LaTeX ($...$)
    def replace_inline(match):
        nonlocal count
        placeholder = f"LATEX_INLINE_{count}_"
        placeholders[placeholder] = match.group(0)
        count += 1
        return placeholder

    text = re.sub(r'(?<!\\)\$(.*?)(?<!\\)\$', replace_inline, text)

    # Process custom image syntax: ![alt|width](src)
    def process_images(match):
        alt_width = match.group(1)
        src = match.group(2)

        # Check if there's a width specification
        if '|' in alt_width:
            alt, width = alt_width.rsplit('|', 1)
            # Try to parse width as number (strip 'px' if present)
            width = width.strip()
            if width.endswith('px'):
                width = width[:-2]

            # If width is a valid number, add the width attribute
            if width.isdigit():
                return f'![{alt}]({src})<!-- img-width:{width} -->'

        # If no width specification or invalid format, return unchanged
        return f'![{alt_width}]({src})'

    # Process images with custom syntax
    text = re.sub(r'!\[(.*?)\]\((.*?)\)', process_images, text)

    # Process with markdown
    html = markdown.markdown(
        text,
        extensions=['fenced_code', 'codehilite', 'tables', 'md_in_html'],
        extension_configs={
            'codehilite': {
                'css_class': 'code-block',  # custom CSS class (in /static/code.css)
            }
        }
    )

    # Post-process HTML to add width attributes to images
    def add_image_width(match):
        img_tag = match.group(1)
        width = match.group(2)

        # If img tag already has a width attribute, don't modify
        if 'width=' in img_tag:
            return f'<img {img_tag}>'

        # Add width attribute
        return f'<img {img_tag} width="{width}">'

    # Add width attributes to images with comments
    html = re.sub(r'<img (.*?)><!-- img-width:(\d+) -->', add_image_width, html)

    # Restore LaTeX parts
    for placeholder, latex in placeholders.items():
        html = html.replace(placeholder, latex)

    return html

# ======================
# External Sync Features
# ======================

def create_blog_index(blog_dir):
    """Create _index.md file for the blog section"""
    index_content = """---
title: Blog
description: My personal blog posts and thoughts
---
"""

    index_path = blog_dir / "_index.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)

    print(f"  ✓ Created blog index: _index.md")

def process_assets(content, source_file, target_dir, source_type):
    """Process assets based on the source type"""
    if source_type == "obsidian":
        return process_obsidian_assets(content, source_file, target_dir)
    else:  # Default to basic markdown
        return process_markdown_assets(content, source_file, target_dir)

def process_markdown_assets(content, source_file, target_dir):
    """Process standard markdown content to find assets and copy them"""
    source_dir = source_file.parent
    assets_dir = target_dir / "assets"
    assets_copied = 0

    # Basic markdown patterns
    asset_patterns = [
        r'!\[([^\]]*)\]\(([^)]+\.[a-zA-Z0-9]+)\)',  # Images: ![alt](file.ext)
        r'(?<!!)\[([^\]]+)\]\(([^)]+\.[a-zA-Z0-9]+)\)'  # Links to files: [text](file.ext)
    ]

    def replace_asset_reference(match, pattern_type):
        nonlocal assets_copied

        if pattern_type == 'image':
            alt_text = match.group(1)
            asset_path = match.group(2)
            prefix = f"![{alt_text}]"
        else:  # link
            link_text = match.group(1)
            asset_path = match.group(2)
            prefix = f"[{link_text}]"

        # Skip URLs
        if asset_path.startswith(('http://', 'https://', '//')):
            return match.group(0)

        # Resolve asset path relative to source file
        if asset_path.startswith('/'):
            full_asset_path = source_dir / asset_path.lstrip('/')
        else:
            full_asset_path = source_dir / asset_path

        # Normalize and check if exists
        try:
            full_asset_path = full_asset_path.resolve()
        except (OSError, ValueError):
            return match.group(0)

        if not full_asset_path.exists():
            print(f"    Warning: Asset not found: {asset_path}")
            return match.group(0)

        # Copy asset
        if not assets_dir.exists():
            ensure_dir(assets_dir)

        asset_filename = full_asset_path.name
        target_asset_path = assets_dir / asset_filename

        # Handle filename conflicts
        counter = 1
        original_stem = full_asset_path.stem
        original_suffix = full_asset_path.suffix

        while target_asset_path.exists():
            asset_filename = f"{original_stem}_{counter}{original_suffix}"
            target_asset_path = assets_dir / asset_filename
            counter += 1

        try:
            shutil.copy2(full_asset_path, target_asset_path)
            assets_copied += 1

            # Convert to img tag if it's an image
            if pattern_type == 'image':
                return f'<img src="assets/{asset_filename}" alt="{alt_text}">'
            else:
                return f"{prefix}(assets/{asset_filename})"
        except (OSError, shutil.Error) as e:
            print(f"    Warning: Failed to copy asset {asset_path}: {e}")
            return match.group(0)

    # Process patterns
    content = re.sub(asset_patterns[0], lambda m: replace_asset_reference(m, 'image'), content)
    content = re.sub(asset_patterns[1], lambda m: replace_asset_reference(m, 'link'), content)

    return content, assets_copied

def find_file_in_vault(filename, source_file):
    """Find a file by name anywhere in the Obsidian vault"""
    # Get the vault root by finding the directory that contains .obsidian
    vault_root = None
    current_dir = source_file.parent

    # Walk up the directory tree to find the vault root
    while current_dir != current_dir.parent:
        if (current_dir / '.obsidian').exists():
            vault_root = current_dir
            break
        current_dir = current_dir.parent

    # If we can't find .obsidian, use the source file's directory as fallback
    if not vault_root:
        vault_root = source_file.parent
        # Also try to find it by going up one more level (in case blog is a subfolder)
        if not (vault_root / filename).exists() and (vault_root.parent / filename).exists():
            vault_root = vault_root.parent

    # Search for the file in the vault
    for file_path in vault_root.glob(f"**/{filename}"):
        # Skip files in .obsidian directory
        if '.obsidian' in file_path.parts:
            continue
        if file_path.is_file():
            return file_path

    # If not found with exact name, try case-insensitive search
    filename_lower = filename.lower()
    for file_path in vault_root.glob("**/*"):
        if file_path.is_file() and file_path.name.lower() == filename_lower:
            # Skip files in .obsidian directory
            if '.obsidian' in file_path.parts:
                continue
            return file_path

    return None

def process_obsidian_assets(content, source_file, target_dir):
    """Process Obsidian content with support for ![[]] syntax and vault-wide file search"""
    source_dir = source_file.parent
    assets_dir = target_dir / "assets"
    assets_copied = 0

    # Obsidian and standard markdown patterns
    asset_patterns = [
        r'!\[\[([^\]]+(?:\|[^\]]*)?)\]\]',  # Obsidian: ![[file|width]] or ![[file]]
        r'!\[([^\]]*)\]\(([^)]+\.[a-zA-Z0-9]+)\)',  # Standard: ![alt](file.ext)
        r'(?<!!)\[([^\]]+)\]\(([^)]+\.[a-zA-Z0-9]+)\)'  # Links: [text](file.ext)
    ]

    def replace_asset_reference(match, pattern_type):
        nonlocal assets_copied

        if pattern_type == 'obsidian':
            path_width = match.group(1)
            if '|' in path_width:
                asset_path, width = path_width.rsplit('|', 1)
                alt_text = f"Image|{width.strip()}"
            else:
                asset_path = path_width
                alt_text = "Image"

            # Search for file in vault
            full_asset_path = find_file_in_vault(asset_path, source_file)
            if not full_asset_path:
                print(f"    Warning: Obsidian asset not found: {asset_path}")
                return match.group(0)

        elif pattern_type == 'image':
            alt_text = match.group(1)
            asset_path = match.group(2)

            # Standard path resolution
            if asset_path.startswith(('http://', 'https://', '//')):
                return match.group(0)

            if asset_path.startswith('/'):
                full_asset_path = source_dir / asset_path.lstrip('/')
            else:
                full_asset_path = source_dir / asset_path

            try:
                full_asset_path = full_asset_path.resolve()
            except (OSError, ValueError):
                return match.group(0)

            if not full_asset_path.exists():
                print(f"    Warning: Asset not found: {asset_path}")
                return match.group(0)

        else:  # link
            link_text = match.group(1)
            asset_path = match.group(2)

            if asset_path.startswith(('http://', 'https://', '//')):
                return match.group(0)

            if asset_path.startswith('/'):
                full_asset_path = source_dir / asset_path.lstrip('/')
            else:
                full_asset_path = source_dir / asset_path

            try:
                full_asset_path = full_asset_path.resolve()
            except (OSError, ValueError):
                return match.group(0)

            if not full_asset_path.exists():
                print(f"    Warning: Asset not found: {asset_path}")
                return match.group(0)

        # Copy asset (common logic)
        if not assets_dir.exists():
            ensure_dir(assets_dir)

        asset_filename = full_asset_path.name
        target_asset_path = assets_dir / asset_filename

        # Handle conflicts
        counter = 1
        original_stem = full_asset_path.stem
        original_suffix = full_asset_path.suffix

        while target_asset_path.exists():
            asset_filename = f"{original_stem}_{counter}{original_suffix}"
            target_asset_path = assets_dir / asset_filename
            counter += 1

        try:
            shutil.copy2(full_asset_path, target_asset_path)
            assets_copied += 1

            if pattern_type == 'obsidian':
                # Convert to img tag, with width if specified
                if '|' in path_width and path_width.rsplit('|', 1)[1].strip().isdigit():
                    width = path_width.rsplit('|', 1)[1].strip()
                    return f'<img src="assets/{asset_filename}" alt="Image" width="{width}">'
                else:
                    return f'<img src="assets/{asset_filename}" alt="Image">'
            elif pattern_type == 'image':
                return f'<img src="assets/{asset_filename}" alt="{alt_text}">'
            else:  # link
                return f"[{link_text}](assets/{asset_filename})"

        except (OSError, shutil.Error) as e:
            print(f"    Warning: Failed to copy asset {asset_path}: {e}")
            return match.group(0)

    # Process patterns
    content = re.sub(asset_patterns[0], lambda m: replace_asset_reference(m, 'obsidian'), content)
    content = re.sub(asset_patterns[1], lambda m: replace_asset_reference(m, 'image'), content)
    content = re.sub(asset_patterns[2], lambda m: replace_asset_reference(m, 'link'), content)

    return content, assets_copied

# =============
# Miscellaneous
# =============

def generate_sitemap(pages, config, public_dir):
    """Generate sitemap.xml for search engines"""
    sitemap_path = Path(public_dir) / "sitemap.xml"
    base_url = config.get("base_url", "")

    sitemap = ['<?xml version="1.0" encoding="UTF-8"?>']
    sitemap.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    for page in pages:
        sitemap.append('  <url>')
        sitemap.append(f'    <loc>{base_url}{page["url"]}</loc>')
        sitemap.append(f'    <lastmod>{page["date"].isoformat()}</lastmod>')
        sitemap.append('    <changefreq>monthly</changefreq>')
        sitemap.append('  </url>')

    sitemap.append('</urlset>')

    with open(sitemap_path, 'w') as f:
        f.write('\n'.join(sitemap))

    print(f"Sitemap generated at {sitemap_path}")

def create_new_post():
    """Create a new blog post with user-provided title"""
    # Ask the user for the post title
    title = input("Enter the title for your new blog post: ").strip()

    if not title:
        print("Error: Title cannot be empty")
        return

    # Create slug from title
    slug = slugify(title)

    # Define paths
    blog_dir = Path("content/blog")
    post_dir = blog_dir / slug

    # Ensure blog directory exists
    if not blog_dir.exists():
        print(f"Creating blog directory at {blog_dir}")
        blog_dir.mkdir(parents=True, exist_ok=True)

    # Check if post directory already exists
    if post_dir.exists():
        print(f"Error: A post with the slug '{slug}' already exists at {post_dir}")
        return

    # Create post directory
    post_dir.mkdir(parents=True, exist_ok=True)

    # Get today's date in YYYY-MM-DD format
    today = datetime.now().strftime("%Y-%m-%d")

    # Create frontmatter content
    frontmatter = f"""---
title: {title}
date: {today}
tags: []
math: false
---

Write your post content here...
"""

    # Create the index.md file
    index_file = post_dir / "index.md"
    with open(index_file, "w") as f:
        f.write(frontmatter)

    print(f"Created new blog post at {index_file}")
    print(f"Post URL will be: /blog/{slug}/")
