import React from 'react'
import { FileTree } from 'nextra/components'

// スタイル定義
const highlightStyle = { color: "#991b1b", textDecoration: "none", fontWeight: "bold" };
const linkStyleBase = { textDecoration: "underline", cursor: "pointer", display: "inline-flex", alignItems: "baseline", gap: "4px" };
const linkDefaultColor = { color: "#0070f3" };

export const ProjectStructure = ({ highlights = [] }: { highlights?: string[] }) => {

    const f = (name: string, url?: string) => {
        const isHighlighted = highlights.includes(name);

        if (url) {
            const finalLinkStyle = {
                ...linkStyleBase,
                ...(isHighlighted ? highlightStyle : linkDefaultColor)
            };
            return (
                <a href={url} target="_blank" rel="noopener noreferrer" style={finalLinkStyle}>
                    {name}
                    <svg fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" height="0.8em" style={{ display: 'inline', verticalAlign: 'baseline' }}>
                        <path d="M7 17L17 7"></path><path d="M7 7h10v10"></path>
                    </svg>
                </a>
            );
        }

        return isHighlighted ? <span style={highlightStyle}>{name}</span> : name;
    };

    // 判定関数
    const hasH = (names: string[]) => names.some(name => highlights.includes(name));

    // 各階層のファイルリスト定義
    const sourcesSubFiles = ["{file_name}.mp4", "BU.pickle", "CTD.csv"];
    const modelsFiles = ["best.pt", "sam_vit_h_4b8939.pth"];
    const sourcesAll = [...sourcesSubFiles, ...modelsFiles];

    const outputSubFiles = ["{threshould}_{filename}_{frames}.mp4", "bees.pkl", "data_graph.pkl", "gt.txt", "trackers.npz"];

    return (
        <FileTree>
            <FileTree.Folder name="KptSort" defaultOpen>

                {/* datasets */}
                <FileTree.Folder name="datasets" defaultOpen={hasH(["train", "val"])}>
                    <FileTree.Folder name="datasets" defaultOpen={hasH(["train", "val"])}>
                        <FileTree.Folder name="images">
                            <FileTree.File name={f("train")} />
                            <FileTree.File name={f("val")} />
                        </FileTree.Folder>
                        <FileTree.Folder name="labels">
                            <FileTree.File name={f("train")} />
                            <FileTree.File name={f("val")} />
                        </FileTree.Folder>
                    </FileTree.Folder>
                </FileTree.Folder>

                <FileTree.Folder name="debug_frames">{""}</FileTree.Folder>

                <FileTree.Folder name="installer" defaultOpen={hasH(["environment.sh", "install_cv2_w_ffmpeg.sh", "requirements.txt"])}>
                    <FileTree.File name={f("environment.sh", "https://github.com/...")} />
                    <FileTree.File name={f("install_cv2_w_ffmpeg.sh", "https://github.com/...")} />
                    <FileTree.File name={f("requirements.txt", "https://github.com/...")} />
                </FileTree.Folder>

                {/* output */}
                <FileTree.Folder name="output" defaultOpen={hasH(outputSubFiles)}>
                    <FileTree.Folder name="{file_name}" defaultOpen={hasH(outputSubFiles)}>
                        <FileTree.File name={f("{threshould}_{filename}_{frames}.mp4")} />
                        <FileTree.File name={f("bees.pkl")} />
                        <FileTree.File name={f("data_graph.pkl")} />
                        <FileTree.File name={f("gt.txt")} />
                        <FileTree.File name={f("trackers.npz")} />
                    </FileTree.Folder>
                </FileTree.Folder>

                {/* sources - BU.pickleが含まれる場合に親も子も開くように修正 */}
                <FileTree.Folder name="sources" defaultOpen={hasH(sourcesAll)}>
                    <FileTree.Folder name="{file_name}" defaultOpen={hasH(sourcesSubFiles)}>
                        <FileTree.File name={f("{file_name}.mp4")} />
                        <FileTree.File name={f("BU.pickle")} />
                        <FileTree.File name={f("CTD.csv")} />
                    </FileTree.Folder>
                    <FileTree.Folder name="Models" defaultOpen={hasH(modelsFiles)}>
                        <FileTree.File name={f("best.pt")} />
                        <FileTree.File name={f("sam_vit_h_4b8939.pth")} />
                    </FileTree.Folder>
                </FileTree.Folder>

                {/* tools */}
                <FileTree.Folder name="tools" defaultOpen={hasH(["AssignBeeHive.py", "calk_oks.py", "generate_graph.py", "kpsort.py", "loadpkl_jit.py"])}>
                    <FileTree.File name={f("AssignBeeHive.py", "/KptSORT/Scripts/tools/AssignBeeHive.py")} />
                    <FileTree.File name={f("calk_oks.py", "/KptSORT/Scripts/tools/calk_oks.py")} />
                    <FileTree.File name={f("generate_graph.py", "/KptSORT/Scripts/tools/generate_graph.py")} />
                    <FileTree.File name={f("kpsort.py", "/KptSORT/Scripts/tools/kpsort.py")} />
                    <FileTree.File name={f("loadpkl_jit.py", "/KptSORT/Scripts/tools/loadpkl_jit.py")} />
                </FileTree.Folder>

                <FileTree.File name={f(".gitignore")} />
                <FileTree.File name={f(".gitlab-ci.yml", "/KptSORT/Scripts/gitlab-ci.yml")} />
                <FileTree.File name={f("analysis.py", "/KptSORT/Scripts/analysis.py")} />
                <FileTree.File name={f("dashboard.py", "/KptSORT/Scripts/dashboard.py")} />
                <FileTree.File name={f("tracking.py", "/KptSORT/Scripts/tracking.py")} />
                <FileTree.File name={f("train.py", "/KptSORT/Scripts/train.py")} />
            </FileTree.Folder>
        </FileTree>
    );
};