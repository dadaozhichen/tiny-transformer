import regex as re 
from typing import BinaryIO
from tqdm import tqdm 
import os  

class TokenizerTrainer():
    def __init__(self,file_path,vacab_size,special_tokens,processor_num) -> None:
        self.file_path = file_path
        self.vocab_size = vacab_size
        self.special_tokens = special_tokens
        self.processor_num = processor_num

        #inistalize vocabulary 
        self.vocab:dict[int,bytes] = {i:bytes([i]) for i in range(256)}
        for i in range(len(special_tokens)):
            self.vocab[256+i] = special_tokens[i].encode('utf-8')

        #initialize merges
        self.merges=[]
        # pre_tokens:{'pre_token_index':[pretoken bytes tuple,count]}   
        self.pre_tokens = {}
        #token->index
        self.pre_token_to_index = {}
        #count the pair of byte pair and store the position of pretoken  bp_freq:dict{(byte1,byte2):[count,[index1,index2,...]]}
        self.bp_freq = {}


    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def train_bpe(self):
        with open(self.file_path,'rb') as f:
            concat_special_token = "|".join(self.special_tokens)
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

            boundaries = self.find_chunk_boundaries(f,self.processor_num,b"<|endoftext|>")

            for start,end in zip(boundaries[:-1],boundaries[1:]):
                f.seek(start)
                chunk = f.read(end-start).decode("utf-8",errors="ignore")

                splited_chunk = re.split(concat_special_token,chunk)

                i=0
                for seg in splited_chunk: #pre tokenization
                    pre_token_iter = re.finditer(PAT,seg)
                    for pre_token in pre_token_iter:
                        pre_token = tuple(pre_token.group().encode("utf-8"))
                        if pre_token not in self.pre_token_to_index:
                            self.pre_token_to_index[pre_token] = i
                            self.pre_tokens[i] = [pre_token,1]
                            i+=1 
                        else:
                            self.pre_tokens[self.pre_token_to_index[pre_token]][1]+=1
                        
            self.init_bp_freq()
            for i in range(self.vocab_size-len(self.vocab)):
                self.merge_bpe()
            
        return self.vocab,self.merges
    

    def init_bp_freq(self): #initialize byte pair frequency
        for pre_token_index,pre_token_data in tqdm(self.pre_tokens.items(),desc="Init bp_freq"):
            pre_token = pre_token_data[0]
            pre_count = pre_token_data[1]
            bytelist = pre_token 
            all_bps = list(zip(bytelist[:-1],bytelist[1:]))

            for bp_i in all_bps:
                if bp_i not in self.bp_freq:
                    self.bp_freq[bp_i] = [0,[]]
                self.bp_freq[bp_i][0]+=pre_count
            
            for bp_i in set(all_bps):
                self.bp_freq[bp_i][1].append(pre_token_index)


    def merge_bpe(self):
        pair = max(self.bp_freq,key=lambda x:(self.bp_freq[x][0],self.vocab[x[0]],self.vocab[x[1]]))  #pair : tuple(str,str) 
        index1,index2 = pair
        new_index = len(self.vocab)
        self.vocab[new_index] = self.vocab[index1] + self.vocab[index2]

        bytes_pair = (self.vocab[index1],self.vocab[index2])
        self.merges.append(bytes_pair)
        self.merge(pair,new_index)


    def merge(self,pair,new_index):
        have_pair_tokens = self.bp_freq[pair][1]
        for have_pair_token_index in have_pair_tokens:
            bytestr = self.pre_tokens[have_pair_token_index]
            new_pre_token = bytestr 

            i=0
            while i<len(new_pre_token)-1:
                slid_window = new_pre_token[i:i+2]
                if pair == slid_window:
                    pre_count = self.pre_tokens[have_pair_token_index][1]
                    if i>0:
                        new_pair = (new_pre_token[i-1],new_index)
                        if new_pair not in self.bp_freq:
                            self.bp_freq[new_pair] = [pre_count,[have_pair_token_index]]
                        else:
                            self.bp_freq[new_pair][0] +=pre_count
                            if not self.bp_freq[new_pair][1] or self.bp_freq[new_pair][1][-1]!=have_pair_token_index:
                                self.bp_freq[new_pair][1].append(have_pair_token_index)
                        
                        overlap_pair = (new_pre_token[i-1],pair[0])
                        self.bp_freq[overlap_pair][0] -= pre_count
                        if self.bp_freq[overlap_pair][0] <=0:
                            self.bp_freq.pop(overlap_pair)
                        
                    if i+2 < len(new_pre_token):
                        new_pair = (new_index,new_pre_token[i+2])
                        if new_pair not in self.bp_freq:
                            self.bp_freq[new_pair] = [pre_count,[have_pair_token_index]]
                        else:
                            self.bp_freq[new_pair][0] += pre_count
                            if not self.bp_freq[new_pair][1] or self.bp_freq[new_pair][1][-1]!=have_pair_token_index:
                                self.bp_freq[new_pair][1].append(have_pair_token_index)
                        overlap_pair = (new_pre_token[i+2],pair[1])
                        self.bp_freq[overlap_pair][0] -= pre_count
                        if self.bp_freq[overlap_pair][0] <= 0:
                            self.bp_freq.pop(overlap_pair)
                    
                    self.bp_freq[pair][0]-=pre_count
                    if self.bp_freq[pair][0]<=0:
                        self.bp_freq.pop(pair)
                    
                    new_pre_token = new_pre_token[:i]+(new_index,)+new_pre_token[i+2:]

                i+=1
            self.pre_tokens[have_pair_token_index][0] = new_pre_token 






