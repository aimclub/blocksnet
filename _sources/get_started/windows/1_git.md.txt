# Install git

Git is required to work with BlocksNet code or to use the library in your own project.

## Download and install

1. Download `git` from [official website](https://git-scm.com/downloads/win) and run the installator.
2. During the installation:
   - Keep the default options.
   - When asked about the editor, choose VS Code or Notepad.
   - Leave the other settings as recommended.
3. Check the version in terminal:
   ```
   git --version
   ```
   The result may be the following:
   ```
   git version 2.33.0.windows.2
   ```

## Clone the repository

If you work with your own repository, clone it instead:

```
git clone https://github.com/aimclub/blocksnet C:\projects\blocksnet
```

Where:

- `https://github.com/aimclub/blocksnet` - the repository to clone.
- `C:\projects\blocksnet` - path to the directory where repository files will be located. Change it according to your preferences.

Then proceed to the cloned repository directory:

```
cd C:\projects\blocksnet
```
