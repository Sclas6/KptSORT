import React from 'react'
import { FileTree } from 'nextra/components'

// スタイル定義
const highlightStyle = { color: "#991b1b", textDecoration: "none", fontWeight: "bold" };
const linkStyleBase = { textDecoration: "underline", cursor: "pointer", display: "inline-flex", alignItems: "baseline", gap: "4px" };
const linkDefaultColor = { color: "#0070f3" };

export const ProjectStructure = ({ highlights = [] }: { highlights?: string[] }) => {

    // ヘルパー関数: urlがあればリンク、なければテキスト(強調判定あり)
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

    // 判定関数: リストの中に指定したファイル名が含まれているか
    const hasH = (names: string[]) => names.some(name => highlights.includes(name));

    return (
        <FileTree>
            <FileTree.Folder name="KptSort" defaultOpen>

                {/* datasets: train, val などが含まれる場合 */}
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

                {/* installer: environment.sh, requirements.txt などが含まれる場合 */}
                <FileTree.Folder name="installer" defaultOpen={hasH(["environment.sh", "install_cv2_w_ffmpeg.sh", "requirements.txt"])}>
                    <FileTree.File name={f("environment.sh", "https://github.com/...")} />
                    <FileTree.File name={f("install_cv2_w_ffmpeg.sh", "https://github.com/...")} />
                    <FileTree.File name={f("requirements.txt", "https://github.com/...")} />
                </FileTree.Folder>

                <FileTree.Folder name="KptSORT-docs" defaultOpen={hasH(["[[...mdxPath]]"])}>
                    <FileTree.Folder name="app">
                        <FileTree.File name={f("[[...mdxPath]]")} />
                    </FileTree.Folder>
                    <FileTree.Folder name="content">{""}</FileTree.Folder>
                    <FileTree.Folder name="patches">{""}</FileTree.Folder>
                    <FileTree.Folder name="public">{""}</FileTree.Folder>
                </FileTree.Folder>

                {/* output: bees.pkl, trackers.npz などが含まれる場合 */}
                <FileTree.Folder name="output" defaultOpen={hasH(["bees.pkl", "trackers.npz", "gt.txt", "data_graph.pkl"])}>
                    <FileTree.Folder name="{file_name}" defaultOpen>
                        <FileTree.File name={f("{threshould}_{filename}_{frames}.mp4")} />
                        <FileTree.File name={f("bees.pkl", "/KptSORT/Start/")} />
                        <FileTree.File name={f("data_graph.pkl")} />
                        <FileTree.File name={f("gt.txt")} />
                        <FileTree.File name={f("trackers.npz")} />
                        {/* 他のファイルは省略（必要に応じてhasHに追加） */}
                    </FileTree.Folder>
                </FileTree.Folder>

                {/* sources: BU.pickle, CTD.csv, best.pt など */}
                <FileTree.Folder name="sources" defaultOpen={hasH(["BU.pickle", "CTD.csv", "best.pt"])}>
                    <FileTree.Folder name="{file_name}">
                        <FileTree.File name={f("{file_name}.mp4")} />
                        <FileTree.File name={f("BU.pickle")} />
                        <FileTree.File name={f("CTD.csv")} />
                    </FileTree.Folder>
                    <FileTree.Folder name="Models" defaultOpen={hasH(["best.pt"])}>
                        <FileTree.File name={f("best.pt")} />
                        <FileTree.File name={f("sam_vit_h_4b8939.pth")} />
                    </FileTree.Folder>
                </FileTree.Folder>

                {/* tools: kpsort.py など */}
                <FileTree.Folder name="tools" defaultOpen={hasH(["AssignBeeHive.py", "calk_oks.py", "generate_graph.py", "kpsort.py", "loadpkl_jit.py"])}>
                    <FileTree.File name={f("AssignBeeHive.py")} />
                    <FileTree.File name={f("calk_oks.py")} />
                    <FileTree.File name={f("generate_graph.py")} />
                    <FileTree.File name={f("kpsort.py")} />
                    <FileTree.File name={f("loadpkl_jit.py")} />
                </FileTree.Folder>

                <FileTree.File name={f(".gitignore")} />
                <FileTree.File name={f(".gitlab-ci.yml")} />
                <FileTree.File name={f("analysis.py")} />
                <FileTree.File name={f("dashboard.py")} />
                <FileTree.File name={f("tracking.py")} />
                <FileTree.File name={f("train.py")} />
            </FileTree.Folder>
        </FileTree>
    );
};